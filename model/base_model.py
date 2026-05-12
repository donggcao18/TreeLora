import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG
from transformers import GenerationConfig

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        self.generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            temperature=self.args.temperature if self.args.do_sample else None,
            top_p=self.args.top_p if self.args.do_sample else None,
            repetition_penalty=self.args.repetition_penalty,
        )
        
        
    def perplexity_evaluation(self, eval_dataloader, device):
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def _task_eval_from_predictions(self, task, sources_sequences, predicted_sequences, ground_truths):
        if task in ['CodeSearchNet', 'CoST', 'KodCode', 'RunBugRun', 'TheVault_Csharp']:
            calc_codebleu = False
        else:
            calc_codebleu = True
        return compute_metrics(
            predicted_sequences,
            ground_truths,
            calc_codebleu=calc_codebleu,
            language=DATASET_TO_OUTPUT_LANG.get(task, None)
        )

    def _generation_model(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _resolve_max_ans_len(self, task_id):
        max_ans_len = getattr(self.args, "max_ans_len", 256)
        if isinstance(max_ans_len, (list, tuple)):
            if len(max_ans_len) == 1:
                return int(max_ans_len[0])
            return int(max_ans_len[task_id])
        if isinstance(max_ans_len, str) and "," in max_ans_len:
            max_ans_len = max_ans_len.split(",")
            if len(max_ans_len) == 1:
                return int(max_ans_len[0])
            return int(max_ans_len[task_id])
        return int(max_ans_len)

    def save_prediction_rows_jsonl(self, prediction_rows, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            for row in prediction_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _ordered_unique_prediction_rows(self, prediction_rows):
        if prediction_rows and all("__index__" in row for row in prediction_rows):
            rows_by_index = {}
            for row in prediction_rows:
                index = int(row["__index__"])
                if index not in rows_by_index:
                    rows_by_index[index] = row
            prediction_rows = [rows_by_index[index] for index in sorted(rows_by_index)]

        return [
            {key: value for key, value in row.items() if key != "__index__"}
            for row in prediction_rows
        ]
    
    def _gather_prediction_rows(self, prediction_rows):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered_rows = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_rows, prediction_rows)
            prediction_rows = [
                row
                for rank_rows in gathered_rows
                for row in rank_rows
            ]

        return self._ordered_unique_prediction_rows(prediction_rows)
    
    def task_generation_evaluation(self, task, test_dataloader, device, max_ans_len=None,
                                   return_predictions=False, prediction_jsonl_path=None):
        self.model.eval()
        generation_model = self._generation_model()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        sample_indices = []

        if max_ans_len is None:
            max_ans_len = self._resolve_max_ans_len(0)
        max_ans_len = int(max_ans_len)

        is_executable = getattr(self.args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            return_predictions = True
            num_return_sequences = int(getattr(self.args, "num_return_sequences", 1))
            top_k = int(getattr(self.args, "top_k", 0))
            generation_kwargs = self.generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config = self.generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=(self.args.global_rank != 0))
        for step, batch in enumerate(test_dataloader):
            batch_indices = batch.pop('indices', None)
            if batch_indices is not None:
                sample_indices.extend(batch_indices.detach().cpu().tolist())

            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id

                generate_ids = generation_model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
            else:
                predicted_sequences += sequences

            if self.args.global_rank == 0:
                progress_bar.update(1)
                description = f"Test step {step}"
                progress_bar.set_description(description, refresh=False)
        progress_bar.close()
        prediction_rows = [
            {
                "source": source,
                "ground-truth": gt,
                "prediction": pred,
            }
            for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
        ]
        if len(sample_indices) == len(prediction_rows):
            for row, index in zip(prediction_rows, sample_indices):
                row["__index__"] = index

        prediction_rows = self._gather_prediction_rows(prediction_rows)
        sources_sequences = [row["source"] for row in prediction_rows]
        ground_truths = [row["ground-truth"] for row in prediction_rows]
        predicted_sequences = [row["prediction"] for row in prediction_rows]

        metrics = {} if is_executable else self._task_eval_from_predictions(task, sources_sequences, predicted_sequences, ground_truths)

        if return_predictions or prediction_jsonl_path is not None:
            if prediction_jsonl_path is not None and self.args.global_rank == 0:
                self.save_prediction_rows_jsonl(prediction_rows, prediction_jsonl_path)
        if return_predictions:
            return metrics, prediction_rows
        return metrics

    def test_all_tasks_and_save_predictions(self):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        prediction_root = os.path.join(
            self.args.output_dir or ".",
            "predictions",
            f"final-{self.__class__.__name__}"
        )
        if self.args.global_rank == 0:
            os.makedirs(prediction_root, exist_ok=True)

        final_metrics = {}
        for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
            print_rank_0(
                f"***** Final testing on task {task_name} after continual training *****",
                self.args.global_rank,
            )
            test_result, prediction_rows = self.task_generation_evaluation(
                task_name,
                test_dataloader,
                device,
                max_ans_len=self._resolve_max_ans_len(task_idx),
                return_predictions=True,
            )
            final_metrics[task_name] = test_result
            print_rank_0(f"[final-test task={task_name}] result: {test_result}", self.args.global_rank)

            if self.args.global_rank == 0:
                safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                with open(prediction_file, "w", encoding="utf-8") as f:
                    json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)

        if self.args.global_rank == 0:
            metrics_file = os.path.join(prediction_root, "metrics_summary.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=2)
            print_rank_0(f"Saved final-test metrics to {metrics_file}", self.args.global_rank)


    def train_one_task(self, task, i_task, epochs):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch.pop('indices', None)
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()


            # Evaluate perplexity on the validation set.
            # print_rank_0(
            #     f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
            #     self.args.global_rank)
            # perplexity = self.perplexity_evaluation(eval_dataloader, device)
            # print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            # self.model.tput_timer.update_epoch_count()
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)
        # self.test_all_tasks_and_save_predictions()

    
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Successfully saving model after round {}'.format(round), self.args.global_rank)

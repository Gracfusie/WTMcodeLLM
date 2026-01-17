import os
import json
import random

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)


def create_detailed_io_log(
    benchmark, combined_results, graded, metadatas, save_eval_results, metrics, output_file
):
    """
    Create a detailed log file containing all inputs, outputs, and scoring information for evaluation.
    
    Args:
        benchmark: List of problem instances
        combined_results: List of (outputs_list, code_list) tuples
        graded: List of graded results (list of booleans per problem)
        metadatas: List of metadata for each problem
        save_eval_results: List of evaluation results
        metrics: Evaluation metrics dictionary
        output_file: Path to save the detailed log
    """
    detailed_log = {
        "summary": {
            "total_problems": len(benchmark),
            "evaluation_metrics": metrics[0] if metrics and len(metrics) > 0 else {},
        },
        "problems": []
    }
    
    for idx, (instance, (outputs_list, code_list), graded_list, metadata_list) in enumerate(
        zip(benchmark, combined_results, graded, metadatas)
    ):
        # Get test cases
        test_cases = instance.public_test_cases + instance.private_test_cases
        
        # Calculate pass rate for this problem
        pass_count = sum(graded_list)
        pass_rate = pass_count / len(graded_list) if graded_list else 0
        
        problem_log = {
            "problem_index": idx + 1,
            "question_id": instance.question_id,
            "question_title": instance.question_title,
            "question_content": instance.question_content,
            "platform": instance.platform.value if hasattr(instance.platform, 'value') else str(instance.platform),
            "difficulty": instance.difficulty.value if hasattr(instance.difficulty, 'value') else str(instance.difficulty),
            "starter_code": instance.starter_code,
            "num_generations": len(code_list),
            "pass_count": pass_count,
            "pass_rate": f"{pass_rate:.2%}",
            "generations": []
        }
        
        # Process each generation
        for gen_idx, (code, output, is_passed, metadata) in enumerate(
            zip(code_list, outputs_list, graded_list, metadata_list)
        ):
            gen_log = {
                "generation_index": gen_idx + 1,
                "code": code,
                "raw_output": output,
                "passed": is_passed,
                "test_case_results": []
            }
            
            # Parse metadata to get test case results
            if metadata:
                try:
                    if isinstance(metadata, str):
                        metadata_dict = json.loads(metadata)
                    else:
                        metadata_dict = metadata
                    
                    # Extract test case information from metadata
                    if "inputs" in metadata_dict:
                        gen_log["test_inputs"] = metadata_dict.get("inputs", [])
                    if "expected" in metadata_dict:
                        gen_log["expected_outputs"] = metadata_dict.get("expected", [])
                    if "output" in metadata_dict:
                        gen_log["actual_outputs"] = metadata_dict.get("output", [])
                    if "error" in metadata_dict:
                        gen_log["error"] = metadata_dict.get("error", "")
                    if "error_code" in metadata_dict:
                        gen_log["error_code"] = metadata_dict.get("error_code", 0)
                    if "error_message" in metadata_dict:
                        gen_log["error_message"] = metadata_dict.get("error_message", "")
                except:
                    gen_log["metadata_raw"] = str(metadata)
            
            # Add test case details
            for test_idx, test_case in enumerate(test_cases):
                test_log = {
                    "test_index": test_idx + 1,
                    "test_type": test_case.testtype.value if hasattr(test_case.testtype, 'value') else str(test_case.testtype),
                    "input": test_case.input,
                    "expected_output": test_case.output,
                }
                
                # Try to extract actual result for this test case from metadata
                if metadata and isinstance(metadata, str):
                    try:
                        metadata_dict = json.loads(metadata)
                        # If we have actual outputs, try to match by index
                        if "output" in metadata_dict and test_idx < len(metadata_dict.get("output", [])):
                            test_log["actual_output"] = metadata_dict["output"][test_idx]
                    except:
                        pass
                
                gen_log["test_case_results"].append(test_log)
            
            problem_log["generations"].append(gen_log)
        
        detailed_log["problems"].append(problem_log)
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_log, f, indent=2, ensure_ascii=False)


def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    
    # Limit problems for generation and evaluation
    original_benchmark_len = len(benchmark)
    if args.evaluate and args.eval_limit is not None:
        eval_limit = args.eval_limit
        print(f"Limiting to first {eval_limit} problems (out of {original_benchmark_len} total)")
        rng_state = random.getstate()
        rng = random.Random(42)
        indices = sorted(rng.sample(range(len(benchmark)), min(eval_limit, len(benchmark))))
        benchmark = [benchmark[i] for i in indices]
        random.setstate(rng_state)
    elif args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    # for i in range(len(combined_results)):
    #     for j in range(len(combined_results[i][1])):
    #         if "def solve()" in combined_results[i][1][j]:
    #             from lcb_runner.utils.extraction_utils import extract_code, LMStyle

    #             combined_results[i][1][j] = extract_code(
    #                 combined_results[i][0][j], LMStyle.Gemini
    #             )
    #             if "\nsolve()" not in combined_results[i][1][j]:
    #                 combined_results[i][1][j] += "\n\nsolve()"

    #                 # combined_results[i][1][j] += "\n\nsolve()"
    #                 print(combined_results[i][1][j])

    if args.evaluate:
        # Benchmark is already limited to EVAL_LIMIT problems at generation stage
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)
        
        # Create detailed input/output log file with all information
        detailed_log_file = output_path.replace(".json", "_detailed_io_log.json")
        create_detailed_io_log(
            benchmark, combined_results, graded, metadatas, 
            save_eval_results, metrics, detailed_log_file
        )
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Total problems evaluated: {len(benchmark)}")
        if metrics and len(metrics) > 0:
            print(f"  Pass@1: {metrics[0].get('pass@1', 'N/A'):.2%}" if 'pass@1' in metrics[0] else f"  Pass@1: N/A")
            print(f"  Pass@5: {metrics[0].get('pass@5', 'N/A'):.2%}" if 'pass@5' in metrics[0] else f"  Pass@5: N/A")
        print(f"  Detailed log saved to: {detailed_log_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

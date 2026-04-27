import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--random_proj",
        type=str,
        default="normal",
        help="The initialization of the projection matrix A."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=10,
        help="The instrinsic dimension of the projection matrix"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--small_model",
        type=str,
        default="/mnt/data/LLMS/meta-llama/Llama-2-13b-chat-hf",
        help="small_model_dir"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Set the alpha if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Set the sigma if the initialization of the projection matrix A is std."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Set the beta if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--large_model",
        type=str,
        default='gpt-4o',
        help="large_model_str."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='llama2-13b',
        help="The model name of the open-source LLM."    
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='/mnt/data/wenjintao/BLADE-main/BLADE-main/train2.csv',
        help="The data path."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='train.json',
        help="The save dir."    
    )
    args = parser.parse_args()
    return args
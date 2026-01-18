import toml
from pathlib import Path
import json
from askme.rtp import TreeNode, SplitMetrics
import matplotlib.pyplot as plt

def get_all_nodes(node: TreeNode) -> list[TreeNode]:
    """Recursively get all nodes in the tree."""
    nodes = [node]
    if node.left is not None:
        nodes.extend(get_all_nodes(node.left))
    if node.right is not None:
        nodes.extend(get_all_nodes(node.right))
    return nodes



def process_file(input_fn: str):
    input_path = Path(input_fn)
    with open(input_path, 'r') as f:
        json_data = f.read()
    loaded_tree = TreeNode.model_validate_json(json_data)
    nodes = get_all_nodes(loaded_tree)
    plt.rcParams.update({'font.size': 8}) 
    figsize=(3,2)
    
    # Plot Input Tokens per Node
    plt.figure(figsize=figsize)
    hist_data = [node.metrics.llm_input_tokens for node in nodes if node.metrics is not None]
    plt.hist(hist_data, bins=30, color='skyblue', edgecolor='black')
    #plt.title('LLM Input Tokens per Node')
    plt.xlabel('LLM Input Tokens')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(input_path.with_name(input_path.stem + '_llm_input_tokens_hist.pdf'))
    plt.close()
    
    # Plot collection size vs. LLM Input Tokens
    plt.figure(figsize=figsize)
    x_data = [len(node.documents) for node in nodes if node.metrics is not None
                and node.metrics.llm_input_tokens is not None]
    y_data = [node.metrics.llm_input_tokens for node in nodes if node.metrics is not None
                and node.metrics.llm_input_tokens is not None]
    y2_data = [node.metrics.llm_output_tokens for node in nodes if node.metrics is not None
                and node.metrics.llm_output_tokens is not None]
    plt.scatter(x_data, y_data, color='orange', alpha=0.7, label='Input')
    plt.scatter(x_data, y2_data, color='blue', alpha=0.7, label='Output')
    #plt.title('Collection Size vs. LLM Input/Output Tokens')
    # Change the x and y ticks to that quantities are divided by 1000
    plt.xticks([x for x in plt.xticks()[0] if x % 1000 == 0], [int(x/1000) for x in plt.xticks()[0] if x % 1000 == 0])
    plt.yticks([y for y in plt.yticks()[0] if y % 1000 == 0], [int(y/1000) for y in plt.yticks()[0] if y % 1000 == 0])
    plt.xlim(0, max(x_data)*1.05)
    plt.ylim(0, max(y_data + y2_data)*1.05)
    plt.xlabel('Collection Size (thousands)')
    plt.ylabel('Tokens (thousands)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(input_path.with_name(input_path.stem + '_collection_size_vs_llm_input_tokens.pdf'))
    plt.close()
    
    # Plot collection size vs. time spent in NLI, LLM, and label propagation
    plt.figure(figsize=figsize)
    x_data = [len(node.documents) for node in nodes if node.metrics is not None
                and node.metrics.nli_time_ms is not None]
    y_nli = [node.metrics.nli_time_ms/1e3 for node in nodes if node.metrics is not None
                and node.metrics.nli_time_ms is not None]
    y_llm = [node.metrics.llm_request_time_ms/1e3 for node in nodes if node.metrics is not None
                and node.metrics.llm_request_time_ms is not None]
    y_lp = [node.metrics.label_propagation_time_ms/1e3 for node in nodes if node.metrics is not None
                and node.metrics.label_propagation_time_ms is not None]
    plt.scatter(x_data, y_nli, color='red', alpha=0.7, label='NLI')
    plt.scatter(x_data, y_llm, color='blue', alpha=0.7, label='LLM')
    plt.scatter(x_data, y_lp, color='green', alpha=0.7, label='LP')
    #plt.title('Collection Size vs. Time Spent')
    plt.xticks([x for x in plt.xticks()[0] if x % 1000 == 0], [int(x/1000) for x in plt.xticks()[0] if x % 1000 == 0])
    plt.xlim(0, max(x_data)*1.05)
    plt.ylim(0, max(y_nli + y_llm + y_lp)*1.05)
    plt.xlabel('Collection Size (thousands)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(input_path.with_name(input_path.stem + '_collection_size_vs_time_spent.pdf'))
    plt.close() 
    
    # Make a histogram of n_attempts
    plt.figure(figsize=figsize)
    hist_data = [node.metrics.n_attempts for node in nodes if node.metrics is not None]
    plt.hist(hist_data, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Number of Attempts per Node')
    plt.xlabel('Number of Attempts')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(input_path.with_name(input_path.stem + '_n_attempts_hist.pdf'))
    plt.close()
    
    # Print total number of tokens through all the nodes
    total_input_tokens = sum([node.metrics.llm_input_tokens for node in nodes if node.metrics is not None])
    total_output_tokens = sum([node.metrics.llm_output_tokens for node in nodes if node.metrics is not None])
    print(f"Total LLM Input Tokens across all nodes: {total_input_tokens}")
    print(f"Total LLM Output Tokens across all nodes: {total_output_tokens}")   
    print(f"Total Tokens across all nodes: {total_input_tokens + total_output_tokens}")
    print(f"Average LLM Input Tokens per node: {total_input_tokens / len(nodes)}")

    # Correlation between llm output tokens and time spent in llm requests
    llm_output_tokens = [node.metrics.llm_output_tokens for node in nodes if node.metrics is not None]
    llm_request_time = [node.metrics.llm_request_time_ms for node in nodes if node.metrics is not None]
    # Calculate the correlation using scipy and get the p-value
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(llm_output_tokens, llm_request_time)
    print(f"Correlation between LLM Output Tokens and LLM Request Time: {corr} (p-value: {p_value})")

def run():
    args = read_input_arguments()
    inputs = args.input

    for input_fn in inputs:
        process_file(input_fn)
    
    
def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trees")


    # Define arguments
    parser.add_argument("--input",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Input file path")

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__=="__main__":
    run()
    
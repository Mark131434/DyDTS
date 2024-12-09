import os
import logging
import torch
import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import AffinityPropagation
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import segeval
import argparse
from tqdm import tqdm
import warnings  
warnings.filterwarnings("ignore")

class DialogueBertModel(nn.Module):
    """BERT-based model for dialogue embedding and similarity calculation."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', margin: float = 0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.margin = margin
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract sentence embeddings from BERT model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
    def get_similarity(self, sent1_emb: torch.Tensor, sent2_emb: torch.Tensor) -> torch.Tensor:
        """Calculate squared Euclidean distance between sentence embeddings."""
        return torch.sum((sent1_emb - sent2_emb) ** 2, dim=1)

    def get_sentence_embeddings(self, sentences: List[str], device: str = 'cuda') -> np.ndarray:
        """Generate embeddings for a list of sentences."""
        inputs = self.tokenizer(sentences, 
                                padding=True, 
                                truncation=True, 
                                max_length=512, 
                                return_tensors='pt')
        encoded = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            sent_emb = self(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            outputs = sent_emb.cpu().numpy()
        return outputs

class AffinityPropagationAlgorithm:
    """Advanced clustering algorithm for dialogue segmentation."""
    
    def __init__(self, position_weight: float = 13.0):
        """
        Initialize the Affinity Propagation algorithm.
        
        Args:
            position_weight (float): Weight for position-based similarity. Defaults to 13.0.
        """
        self.position_weight = position_weight
        self.logger = logging.getLogger(__name__)
    
    def get_cluster_centers_indices(self, inputs: np.ndarray) -> List[int]:
        """
        Perform clustering using Affinity Propagation with content and position similarity.
        
        Args:
            inputs (np.ndarray): Input embeddings for clustering
        
        Returns:
            List[int]: Indices of cluster centers
        """
        try:
            content_similarity = -euclidean_distances(inputs, squared=True)
            
            positions = np.array([i for i in range(len(inputs))]).reshape(-1, 1)
            position_similarity = -self.position_weight * euclidean_distances(positions, squared=True)
            
            similarity_matrix = content_similarity + position_similarity
            
            ap = AffinityPropagation(
                preference=np.min(content_similarity), 
                affinity='precomputed', 
                random_state=1
            ).fit(similarity_matrix)
            
            cluster_centers_indices = list(ap.cluster_centers_indices_)
            
            # Merge adjacent indices and handle boundary cases
            cluster_centers_indices = self._merge_adjacent(cluster_centers_indices)
            cluster_centers_indices = [idx for idx in cluster_centers_indices if 0 < idx < len(inputs)]
            
            # self.logger.info(f"Cluster centers indices: {cluster_centers_indices}")
            return cluster_centers_indices
        
        except Exception as e:
            self.logger.error(f"Error in clustering: {e}")
            return []
    
    def _merge_adjacent(self, indices: List[int]) -> List[int]:
        """
        Merge adjacent indices to reduce fragmentation.
        
        Args:
            indices (List[int]): List of cluster center indices
        
        Returns:
            List[int]: Merged indices
        """
        if len(indices) <= 1:
            return indices
        
        result = [indices[0]]
        for idx in indices[1:]:
            if idx - result[-1] > 1:
                result.append(idx)
        
        return result

def load_model_for_inference(
    model: DialogueBertModel, 
    checkpoint_path: Optional[str] = None, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> DialogueBertModel:
    """
    Load model checkpoint and prepare for inference.
    
    Args:
        model (DialogueBertModel): Model to load
        checkpoint_path (str, optional): Path to checkpoint file
        device (str): Device to load model on
    
    Returns:
        DialogueBertModel: Model prepared for inference
    """
    if checkpoint_path and os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def adjust_boundaries_within_contexts(embeddings: np.ndarray, initial_boundaries: List[int]) -> List[int]:
    """
    Adjust segment boundaries based on local similarity.
    
    Args:
        embeddings (np.ndarray): Sentence embeddings
        initial_boundaries (List[int]): Initial segment boundaries
    
    Returns:
        List[int]: Refined segment boundaries
    """
    initial_boundaries = [0] + initial_boundaries + [len(embeddings)]
    contexts = [(initial_boundaries[i], initial_boundaries[i+2]) for i in range(len(initial_boundaries) - 2)]
    
    new_boundaries = []
    new_boundary = 0
    
    for start, end in contexts:
        if new_boundary > 0:
            start = new_boundary
        
        segment_embeddings = embeddings[start:end]
        similarity_matrix = cosine_similarity(segment_embeddings)
        
        # Calculate pairwise similarities between adjacent sentences
        pairwise_similarities = [similarity_matrix[i, i + 1] for i in range(len(segment_embeddings) - 1)]
        
        # Find the position with lowest similarity as new boundary
        min_similarity_idx = np.argmin(pairwise_similarities)
        
        new_boundary = start + min_similarity_idx + 1
        new_boundaries.append(new_boundary)
    
    return new_boundaries

def calculate_variance(old_center: List[int], new_center: List[int]) -> float:
    """
    Calculate variance between old and new cluster centers.
    
    Args:
        old_center (List[int]): Previous cluster centers
        new_center (List[int]): New cluster centers
    
    Returns:
        float: Squared distance between centers
    """
    old_center = np.array(old_center)
    new_center = np.array(new_center)
    squared_distances = np.sum((old_center - new_center) ** 2, axis=-1)
    return squared_distances

def convert_to_binary_segments(contents: List[str], seg_points: List[int]) -> List[int]:
    """
    Convert segmentation points to segment lengths.
    
    Args:
        contents (List[str]): Original text contents
        seg_points (List[int]): Segmentation points
    
    Returns:
        List[int]: Segment lengths
    """
    results_p = []
    seg_p_labels = [0]*(len(contents)+1)
    for i in seg_points:
        seg_p_labels[i] = 1

    tmp = 0
    for fake in seg_p_labels:
        if fake == 1:
            tmp+=1
            results_p.append(tmp)
            tmp = 0
        else:
            tmp += 1
    results_p.append(tmp)
    results_p[0] = results_p[0] -1
    return results_p

def main_process_plus(contents: List[str], model: DialogueBertModel, position_weight: float) -> List[int]:
    """
    Main process for dialogue segmentation with iterative refinement.
    
    Args:
        contents (List[str]): Dialogue contents
        model (DialogueBertModel): Embedding model
    
    Returns:
        List[int]: Final segmentation points
    """
    embeddings = model.get_sentence_embeddings(contents)
    
    # Initialize affinity propagation algorithm
    ap_algorithm = AffinityPropagationAlgorithm(position_weight)
    init_seg = ap_algorithm.get_cluster_centers_indices(embeddings)
    
    variance = 100
    iteration = 0
    
    while variance > 2:
        new_seg = adjust_boundaries_within_contexts(embeddings, init_seg)
        variance = calculate_variance(init_seg, new_seg)
        
        init_seg = new_seg
        iteration += 1
        
        if iteration > 5:
            break
    
    return init_seg

def evaluate_segmentation(input_path: str, model: DialogueBertModel, position_weight: float):
    """
    Evaluate dialogue segmentation on a dataset.
    
    Args:
        input_path (str): Path to input documents
        model (DialogueBertModel): Embedding model
    """
    input_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    window_diff_scores = []
    pk_scores = []
    total_turns = 0
    error_count = 0
    
    for file_name in tqdm(input_files,desc="Reading dialogues"):
        try:
            file_path = os.path.join(input_path, file_name)
            index = []
            contents = []
            tmp = 0
            
            with open(file_path, 'r') as file:
                for line in file:
                    if '================' in line.strip():
                        index.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                        contents.append(line)
                index.append(tmp)
            
            # Perform segmentation
            seg_predicted = main_process_plus(contents, model, position_weight)
            seg_reference = index
            
            # Convert to segment lengths
            seg_p = convert_to_binary_segments(contents, seg_predicted)
            
            # print(f"Predicted Segments: {seg_p}")
            # print(f"Reference Segments: {seg_reference}")
            
            total_turns += len(seg_reference)
            
            # Evaluate segmentation
            window_diff_scores.append(segeval.window_diff(seg_p, seg_reference))
            pk_scores.append(segeval.pk(seg_p, seg_reference))
        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            error_count += 1
    
    # Calculate average scores
    avg_window_diff = np.mean(window_diff_scores)
    avg_pk = np.mean(pk_scores)
    
    print(f"Average Window Diff Score: {avg_window_diff}")
    print(f"Average Pk Score: {avg_pk}")
    print(f"Total Errors: {error_count}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据相关参数
    parser.add_argument("--model_name", type=str, required=True,
                      help="Directory containing the pkl data files")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to dialogue data")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to checkpoint for inference")
    parser.add_argument("--position_weight", type=float, default=0.01,
                      help="position weight of AP algorithm")
    

    args = parser.parse_args()

    return args
def main():
    """
    Main execution function for dialogue segmentation.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    # Load model
    args = parse_args()
    model = DialogueBertModel(model_name=args.model_name)
    model = load_model_for_inference(
        model, 
        args.checkpoint_path
    )
    # Set input path
    input_path = args.data_path

    position_weight = args.position_weight
    
    # Evaluate segmentation
    evaluate_segmentation(input_path, model, position_weight)

if __name__ == "__main__":
    main()
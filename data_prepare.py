import os
import glob
import pickle
import random
import logging
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

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
    
    def adjust_boundaries_within_contexts(self, embeddings:np.ndarray, initial_boundary:List[int]) -> List[int]:
        initial_boundaries = [0] + initial_boundary + [len(embeddings)]
        contexts = [(initial_boundaries[i],initial_boundaries[i+2]) for i in range(len(initial_boundaries) -2)]
        new_boundaries = []
        new_boundary = 0
        for start, end in contexts:
            if new_boundary>0:
                start = new_boundary
            segment_embeddings = embeddings[start:end]
            similarity_matrix = cosine_similarity(segment_embeddings)

            pairwise_similarities = [similarity_matrix[i, i + 1] for i in range(len(segment_embeddings) - 1)]
            
            min_similarity_idx = np.argmin(pairwise_similarities)
            new_boundary = start + min_similarity_idx + 1 

            if not new_boundary:
                return initial_boundary
            
            new_boundaries.append(new_boundary)
        return new_boundaries
    
    def get_final_segmentation(
        self, 
        embeddings: np.ndarray, 
        max_iterations: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Iteratively refine dialogue segmentation.
        
        Args:
            embeddings (np.ndarray): Sentence embeddings
            max_iterations (int): Maximum refinement iterations
        
        Returns:
            List[Tuple[int, int]]: Final segment boundaries
        """
        current_seg = self.get_cluster_centers_indices(embeddings)
        variance = float('inf')
        iteration = 0
        
        while variance > 2 and iteration < max_iterations:
            new_seg = self.adjust_boundaries_within_contexts(embeddings, current_seg)
            variance = np.sum((np.array(current_seg) - np.array(new_seg)) ** 2)
            current_seg = new_seg
            iteration += 1
        
        # Add start and end boundaries
        current_seg = self._merge_adjacent(current_seg)
        full_boundaries = [0] + current_seg + [len(embeddings)]
        segments = [
            (full_boundaries[i], full_boundaries[i+1]) 
            for i in range(len(full_boundaries) - 1)
        ]
        
        return segments

@dataclass
class DialogueExample:
    dialogue_id: str
    sentences: List[str]
    embeddings: np.ndarray
    core_indices: List[tuple]  

@dataclass
class TrainingSample:
    anchor_idx: int
    positive_indices: List[int]
    hard_negative_indices: List[int]
    regular_negative_indices: List[int]

class DialogueBertModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', margin: float = 0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.margin = margin

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
    def get_similarity(self, sent1_emb: torch.Tensor, sent2_emb: torch.Tensor) -> torch.Tensor:
        return torch.sum((sent1_emb - sent2_emb) ** 2, dim=1)

def load_model_for_inference(
    model: DialogueBertModel, 
    checkpoint_path: Optional[str] = None, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> DialogueBertModel:
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

class DialoguePreprocessor:
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        tokenizer_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        position_weight: float = 13.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        hard_nagative: int = 3,
        ragular_negative: int = 6,
    ):
        """
        Initialize preprocessor with configurable parameters.
        
        Args:
            model_name (str): BERT model name
            tokenizer_path (Optional[str]): Path to tokenizer
            checkpoint_path (Optional[str]): Path to model checkpoint
            position_weight (float): Weight for position-based similarity
            device (str): Device for processing
            hard_nagative: the number of hard negatives
            regular_negative: the number of regular negatives
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path or model_name
        )

        # Initialize model
        self.model = DialogueBertModel(model_name)
        self.model = load_model_for_inference(
            self.model, 
            checkpoint_path, 
            device
        )

        # Set parameters
        self.device = device
        self.ap_algorithm = AffinityPropagationAlgorithm(
            position_weight=position_weight
        )

        self.hard_nagative = hard_nagative
        self.ragular_negative = ragular_negative

        
    def read_dialogues(self, data_dir: str) -> List[Tuple[str, List[str]]]:
        dialogue_files = glob.glob(os.path.join(data_dir, "*.txt"))
        dialogues = []
        
        for file_path in tqdm(dialogue_files, desc="Reading dialogues"):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and '=======' not in line]
                dialogue_id = os.path.basename(file_path).split('.')[0]
                dialogues.append((dialogue_id, lines))
                
        return dialogues
    
    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        inputs = self.tokenizer(sentences, 
                                padding=True, 
                                truncation=True, 
                                max_length=512, 
                                return_tensors='pt')
        encoded = {k:v.to(self.device) for k,v in inputs.items()}
            
        with torch.no_grad():
            sent1_emb = self.model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            outputs = sent1_emb.cpu()
            outputs = outputs.numpy()
        return outputs
    
    def find_core_sentences(self, embeddings: np.ndarray, ap_algorithm: AffinityPropagationAlgorithm) -> List[int]:
        cluster_indices = ap_algorithm.get_final_segmentation(embeddings)
        return cluster_indices
    
    def create_training_samples(self, dialogues: List[str], dialogue_id: int, example: DialogueExample) -> List[TrainingSample]:
        training_samples = []
        total_indecs = list(range(len(dialogues)))
        indices = [item for item in total_indecs if item != dialogue_id]

        for i, (start,end) in enumerate(example.core_indices):
            for j in range(start,end-1):

                hard_negatives = []
                if i > 0:
                    hard_negatives.extend(list(range(example.core_indices[i-1][0],example.core_indices[i-1][1]-1)))
                if i < len(example.core_indices) -1:
                    hard_negatives.extend(list(range(example.core_indices[i+1][0]+2,example.core_indices[i+1][1])))
                
                hard_negative_samples = random.sample(hard_negatives, min(self.hard_nagative,len(hard_negatives)))

                other_negatives = []
                for m, (other_start,other_end) in enumerate(example.core_indices):
                    if abs(i-m) > 1:
                        other_negatives.extend(list(range(other_start,other_end)))

                random_negative_samples = random.sample(other_negatives,min(self.ragular_negative -len(hard_negative_samples),len(other_negatives)))

                random_negative_samples = [example.sentences[item] for item in random_negative_samples]
                hard_negative_samples = [example.sentences[item] for item in hard_negative_samples]

                random_index = random.choice(indices)
                selected_item = random.sample(dialogues[random_index][1], 2)
                random_negative_samples += selected_item
   
                training_samples.append(TrainingSample(
                    anchor_idx=example.sentences[j],
                    positive_indices=example.sentences[j+1],
                    hard_negative_indices=hard_negative_samples,
                    regular_negative_indices=random_negative_samples
                ))
        return training_samples
   
    
    def process_dialogues(self, data_dir: str, output_file: str):
        """处理所有对话并保存训练数据"""
        # 读取对话
        dialogues = self.read_dialogues(data_dir)
        processed_data = []
        ap_algorithm = AffinityPropagationAlgorithm(position_weight=13)
        
        for id,(dialogue_id, sentences) in enumerate(tqdm(dialogues, desc="Processing dialogues")):
            # 获取句子embeddings
            embeddings = self.get_sentence_embeddings(sentences)
            
            # 找到核心句子
            core_indices = self.find_core_sentences(embeddings,ap_algorithm)
            
            # 创建对话示例
            example = DialogueExample(
                dialogue_id=dialogue_id,
                sentences=sentences,
                embeddings=embeddings,
                core_indices=core_indices
            )
            
            # 创建训练样本
            training_samples = self.create_training_samples(dialogues, dialogue_id, example)
            
            processed_data.append({
                'example': example,
                'training_samples': training_samples
            })
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        self.logger.info(f"Processed {len(dialogues)} dialogues")
        self.log_statistics(processed_data)
    
    def log_statistics(self, processed_data: List[Dict]):
        """记录数据统计信息"""
        total_core_sentences = 0
        total_training_samples = 0
        hard_negative_samples_dist = []
        regular_negative_samples_dist = []
        positive_samples_dist = []
        
        for data in processed_data:
            example = data['example']
            training_samples = data['training_samples']
            
            total_core_sentences += len(example.core_indices)
            total_training_samples += len(training_samples)
            
            for sample in training_samples:
                if sample.positive_indices:
                    positive_samples_dist.append(1)
                hard_negative_samples_dist.append(len(sample.hard_negative_indices))
                regular_negative_samples_dist.append(len(sample.regular_negative_indices))
        
        self.logger.info(f"avg segment sentences: {np.mean(total_core_sentences)}")
        self.logger.info(f"Total training samples: {total_training_samples}")
        self.logger.info(f"Average positive samples per anchor: {sum(positive_samples_dist)/total_training_samples:.2f}")
        self.logger.info(f"Average hard negative samples per anchor: {np.mean(hard_negative_samples_dist):.2f}")
        self.logger.info(f"Average regular negative samples per anchor: {np.mean(regular_negative_samples_dist):.2f}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dialogue preprocessing.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Dialogue Preprocessor for Training Data Generation'
    )
    
    # Input and Output Paths
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help='Directory containing dialogue text files'
    )
    parser.add_argument(
        '--file_name', 
        type=str, 
        required=True, 
        help='File name to save processed dialogue data'
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Path to save processed dialogue data")
    
    # Model Configuration
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='bert-base-uncased', 
        help='Pretrained BERT model name'
    )
    parser.add_argument(
        '--tokenizer_path', 
        type=str, 
        help='Path to custom tokenizer'
    )
    parser.add_argument(
        '--checkpoint_path', 
        type=str, 
        help='Path to model checkpoint'
    )
    
    # Preprocessing Parameters
    parser.add_argument(
        '--position_weight', 
        type=float, 
        default=13.0, 
        help='Weight for position-based similarity'
    )

    parser.add_argument(
        '--hard_nagative', 
        type=int, 
        default=3, 
        help='the number of hard negatives'
    )

    parser.add_argument(
        '--ragular_negative', 
        type=int, 
        default=6, 
        help='the number of ragular negatives'
    )
    
    # Device Configuration
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        help='Device for processing (cuda/cpu)'
    )
    
    # Logging
    parser.add_argument(
        '--log_level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging based on argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create preprocessor with parsed arguments
    preprocessor = DialoguePreprocessor(
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        position_weight=args.position_weight,
        device=args.device,
        hard_nagative = args.hard_nagative,
        ragular_negative = args.ragular_negative
    )
    
    # Process dialogues
    file_path = os.path.join(args.output_dir,args.file_name)
    preprocessor.process_dialogues(args.data_dir, file_path)

if __name__ == "__main__":
    main()
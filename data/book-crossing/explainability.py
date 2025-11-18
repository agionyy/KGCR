"""
@Author: Guo Junyan
@Description: Explainability experiment
@Update: 2025.11.14
"""

import torch
import pickle
import numpy as np
from scipy.sparse import load_npz
from collections import deque, defaultdict
from tqdm import tqdm
import random

# Custom modules
try:
    from modules.KGCR import KGCR
    from utils.parser import parse_args_kgsr
except ImportError:
    print("Required modules not found: modules.KGCR or utils.parser")
    exit(1)


class ProductRecommender:
    """
    Basic wrapper to load model and prepare lookup tables.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.model = None
        self.n_users = None
        self.n_items = None
        self.mean_mat = None

        # Caches for explainability
        self.g_scores = None                # (h, r, t) -> score
        self.graph_bwd = None               # t -> [(r, h)]

    def load_model(self):
        """
        Load model state, graph structure, and interaction matrix.
        """

        print("Loading data files...")
        try:
            with open('n_params.pkl', 'rb') as f:
                n_params = pickle.load(f)
            with open('graph.pkl', 'rb') as f:
                graph = pickle.load(f)
            mean_mat = load_npz('mean_mat.npz')

            print("Loading checkpoint...")
            state_dict = torch.load('book-crossing_epoch_75.ckpt', map_location=self.device)

            args = parse_args_kgsr()

            print("Initializing model...")
            model = KGCR(n_params, args, graph, mean_mat)
            model.to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            # Save references
            self.model = model
            self.mean_mat = mean_mat
            self.n_users = n_params["n_users"]
            self.n_items = n_params["n_items"]

            # Build lookup for g-scores and backward adjacency
            print("Preparing g_scores and backward graph...")
            raw_scores, raw_index, raw_type = self.model.generate_global_attn_score()

            self.g_scores = {}
            self.graph_bwd = defaultdict(list)

            idx = raw_index.cpu()
            rel = raw_type.cpu()
            scr = raw_scores.cpu()

            for i in tqdm(range(len(scr)), desc="Preprocessing graph"):
                h = idx[0, i].item()
                t = idx[1, i].item()
                r = rel[i].item()
                s = scr[i].item()

                self.g_scores[(h, r, t)] = s
                self.graph_bwd[t].append((r, h))

            print("Model & graph pre-processing done.")

        except FileNotFoundError as e:
            print("Missing required data file:", e)
            exit(1)

    @torch.no_grad()
    def get_model_recommendations_for_user(self, user_id, history_list, k=20):
        """
        Compute top-K recommended items for a user.
        History items are masked.
        """
        user_emb, item_emb = self.model.generate()
        u_emb = user_emb[user_id].unsqueeze(0).to(self.device)

        # Cosine or dot product depending on model
        scores = (u_emb @ item_emb.to(self.device).T).squeeze(0)

        # Remove already-interacted items
        if history_list:
            scores[history_list] = -1e9

        _, topk_idx = torch.topk(scores, k=k)
        return topk_idx.tolist()


class ExplainabilityExperiment:
    """
    Quantitative explainability analysis using backward-path search.
    """

    def __init__(self, recommender, max_L=3, k=20):
        self.recommender = recommender
        self.mean_mat = recommender.mean_mat
        self.n_items = recommender.n_items
        self.g_scores = recommender.g_scores
        self.graph_bwd = recommender.graph_bwd

        self.max_L = max_L
        self.k = k

        if not self.g_scores or not self.graph_bwd:
            raise RuntimeError("Graph or score cache not prepared correctly.")

        print(f"Experiment initialized: L={max_L}, K={k}")

    def get_top_active_users(self, k=100):
        """
        Return top-K users with highest interaction count.
        """
        user_activity = self.mean_mat.getnnz(axis=1)
        return np.argsort(user_activity)[-k:][::-1]

    def get_user_history_set(self, user_id):
        """
        Return the set of items the user has interacted with.
        """
        return set(self.mean_mat.getrow(user_id).nonzero()[1])

    def get_random_recommendations(self, history_set, k=20):
        """
        Random items not in user's history.
        """
        candidates = list(set(range(self.n_items)) - history_set)
        return random.sample(candidates, k)

    def find_best_path_score_bfs(self, start_item, history_set):
        """
        Backward BFS from a candidate item.
        Averages g-scores along each possible path and keeps the best.
        """

        if start_item in history_set:
            return 0.0

        best_score = 0.0
        queue = deque([(start_item, 0, [])])
        visited = {(start_item, 0)}

        while queue:
            node, depth, path_triples = queue.popleft()

            # If we reached a known item, evaluate this path
            if node in history_set and path_triples:
                path_scores = [self.g_scores.get(t, 0) for t in path_triples]
                best_score = max(best_score, np.mean(path_scores))

            if depth >= self.max_L:
                continue

            # Expand backward neighbors
            if node in self.graph_bwd:
                for r, h in self.graph_bwd[node]:
                    if (h, depth + 1) not in visited:
                        visited.add((h, depth + 1))
                        triple = (h, r, node)
                        queue.append((h, depth + 1, path_triples + [triple]))

        return best_score

    def run_experiment(self):
        """
        Main loop over top users.
        """
        users = self.get_top_active_users(k=100)
        model_scores, random_scores = [], []

        print("Running experiment on Top-100 active users...")

        for u in tqdm(users, desc="Users"):
            history = self.get_user_history_set(u)
            if not history:
                continue

            # Model recommendations
            model_items = self.recommender.get_model_recommendations_for_user(
                u, list(history), k=self.k
            )
            # Random recommendations
            random_items = self.get_random_recommendations(history, k=self.k)

            # Evaluate path quality
            ms = [self.find_best_path_score_bfs(it, history) for it in model_items]
            rs = [self.find_best_path_score_bfs(it, history) for it in random_items]

            model_scores.append(np.mean(ms))
            random_scores.append(np.mean(rs))

        print("\n===== Final Result =====")
        print(f"Model Avg Score : {np.mean(model_scores):.4f}")
        print(f"Random Avg Score: {np.mean(random_scores):.4f}")


if __name__ == "__main__":
    recommender = ProductRecommender()
    recommender.load_model()

    experiment = ExplainabilityExperiment(recommender, max_L=3, k=20)
    experiment.run_experiment()

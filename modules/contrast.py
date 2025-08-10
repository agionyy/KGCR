import torch
import torch.nn.functional as F

class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7, beta: float = 0.1):
        super(Contrast, self).__init__()
        self.tau: float = tau
        self.beta: float = beta

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))

        return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

    # Hard Negative Sampling
    # def loss(self, z1: torch.Tensor, z2: torch.Tensor):
    #     f = lambda x: torch.exp(x / self.tau)
    #
    #     # 计算正样本的相似度
    #     pos_sim = self.self_sim(z1, z2)  # [B]
    #     pos_exp = f(pos_sim)  # 正样本的权重 exp(sim)
    #
    #     # 计算所有负样本的相似度矩阵
    #     sim_matrix = self.sim(z1, z2)  # [B, B]
    #
    #     # mask 掉正样本的相似度
    #     neg_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=z1.device)
    #     neg_sim_matrix = sim_matrix[neg_mask].view(z1.size(0), -1)  # [B, B-1]
    #
    #     # 选出与正样本最相似的负样本（最大相似度）
    #     hardest_neg_sim, _ = neg_sim_matrix.max(dim=1)  # [B]
    #     hardest_neg_exp = f(hardest_neg_sim)  # 计算困难负样本的 exp(sim)
    #
    #     # 计算对比损失
    #     loss = -torch.log(pos_exp / (pos_exp + hardest_neg_exp))
    #     return loss

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor):
        p = F.log_softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        return F.kl_div(p, q, reduction='batchmean')

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        # loss = self.loss(h1, h2).mean()

        contrastive_loss = self.loss(h1, h2).mean()
        kl_loss = self.kl_divergence(h1, h2)
        loss = contrastive_loss + self.beta * kl_loss
        return loss

import torch
from torch import nn
from models.utils.metrics import L2_loss
from models.utils.helper import truncated_normal_


class JNSKR(nn.Module):
    def __init__(self, n_users, n_items, n_relations, n_entities, max_i_u, max_i_r,
                 negative_c, negative_ck, args):
        """
        JNSKR has 8 trainable params: pre_vec, W_att, Bias_att, H_att, User, Item, Entity, Relation
        :param n_users: number of users in all data
        :param n_items:
        :param n_relations:
        :param n_entities:
        :param max_i_u: Maximum number of interactions between item to users
        :param max_i_r: Maximum number of connections between item (head entity) and relations
        :param negative_c:
        :param negative_ck:
        :param args: other arguments
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities

        # Maximum number of interactions between item to users
        self.max_i_u = max_i_u
        # Maximum number of connections between item (head entity) and relations
        self.max_i_r = max_i_r

        self.emb_size = args.embed_size
        self.attention_size = self.emb_size // 2

        # weight of pos & neg samples in KG, respectively
        self.negative_c = negative_c
        self.negative_ck = negative_ck

        # weight of pos and neg samples in CF, respectively
        self.c_v_pos = 1.0
        self.c_v_neg = 0.2

        # weight of multi-tasks
        self.weight_cf_loss = args.coefficient[0]
        self.weight_kg_loss = args.coefficient[1]
        # weight of L2 Regularization
        self.weight_cf_l2 = args.lambda_bilinear[0]
        self.weight_kg_l2 = args.lambda_bilinear[1]

        # Dropout
        self.dropout_kg = nn.Dropout(p=args.dropout_kg, inplace=False)
        self.dropout_cf = nn.Dropout(p=args.dropout_cf, inplace=False)

        # embedding layers (DistMult embed relations as vectors)
        self.User = nn.Embedding(self.n_users + 1, self.emb_size)
        self.Item = nn.Embedding(self.n_items + 1, self.emb_size)
        self.Entity = nn.Embedding(self.n_entities + 1, self.emb_size)
        self.Relation = nn.Embedding(self.n_relations + 1, self.emb_size)

        # h in Equ (9), prediction vector
        self.pre_vec = nn.Parameter(
            torch.full([self.emb_size, 1], fill_value=0.01).double(),
            requires_grad=True
        )

        # >>> attention
        # different from Equ (11), only use ONE matrix, W
        self.W_att = nn.Parameter(
            truncated_normal_(
                torch.empty(size=[self.emb_size, self.attention_size]),
                mean=0.0,
                std=torch.sqrt(torch.div(2.0, self.attention_size + self.emb_size))
            ),
            requires_grad=True
        )
        # b in Equ (11)
        self.Bias_att = nn.Parameter(
            torch.zeros([self.attention_size], dtype=torch.float64), requires_grad=True
        )
        # h_{\alpha} in Equ (11)
        self.H_att = nn.Parameter(
            torch.full([self.attention_size, 1], 0.01).double(), requires_grad=True
        )
        # <<< attention
        self.activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Embedding):
                truncated_normal_(tensor=layer.weight, mean=0.0, std=0.01)

    def cal_att_weight(self, pos_r, pos_t, mask):
        """ calculate attention weight
        """
        # Equ (11) upper part, use ONE matrix, W
        score = self.activation(torch.einsum("abc, ck->abk", pos_r * pos_t, self.W_att) + self.Bias_att)
        score = torch.exp(torch.einsum("abc, ck->abk", score, self.H_att))
        # mask
        score = torch.einsum("ab, abc->abc", mask, score)
        # Equ (11)
        score_sum = torch.sum(score, dim=1, keepdim=True)
        weight = torch.div(score, score_sum)
        return weight

    def cal_loss(self, g_hrt, item_batch_ev, item_batch_qv, y_uv, c, ck):
        """
        :param g_hrt: scores of positive triples (h, r, t)
        :param item_batch_ev: items' embedding vectors
        :param item_batch_qv: items' final embeddings (introduce neighbors' information)
        :param y_uv: prediction of user-item preference
        :param c:
        :param ck:
        :returns total loss
        """
        # KG loss
        # Equ (5), loss for positive data
        loss_kg = torch.sum((1.0 - ck) * torch.pow(g_hrt, 2) - 2.0 * g_hrt)
        # Equ (8), loss for all data
        loss_kg += torch.sum(
            torch.einsum("ab,ac->bc", self.Relation.weight, self.Relation.weight)
            * torch.einsum("hi,hj->ij", ck * item_batch_ev, item_batch_ev)
            * torch.einsum("ti,tj->ij", self.Entity.weight, self.Entity.weight)
        )

        # CF loss
        # Equ (12)
        loss_cf = torch.sum(
            (self.c_v_pos - c) * torch.pow(y_uv, 2) - 2.0 * self.c_v_pos * y_uv
        )
        loss_cf += torch.sum(
            torch.matmul(self.pre_vec, self.pre_vec.t())
            * torch.einsum("ui,uj->ij", self.User.weight, self.User.weight)
            * torch.einsum("vi,vj->ij", c * item_batch_qv, item_batch_qv)
        )

        # L2 Regularization
        l2_kg = torch.tensor(0., requires_grad=True) + L2_loss(self.User.weight) + L2_loss(self.Item.weight) \
                + L2_loss(self.Entity.weight) + L2_loss(self.Relation.weight)
        l2_cf = torch.tensor(0., requires_grad=True) + L2_loss(self.W_att) + L2_loss(self.H_att)
        l2_loss = self.weight_kg_l2 * l2_kg + self.weight_cf_l2 * l2_cf

        # weighted loss
        tot_loss = self.weight_kg_loss * loss_kg + self.weight_cf_loss * loss_cf + l2_loss
        return tot_loss

    def cf_score(self, pos_items, users, r_test, t_test):
        """ predict user-item preference
        :param pos_items:
        :param users:
        :param r_test: relations
        :param t_test: tail entities
        :return:
        """
        r_emb = self.Relation(r_test)
        t_emb = self.Entity(t_test)

        # mask
        pos_num_r = torch.not_equal(r_test, self.n_relations).float()
        # add this line ↓
        r_emb = torch.einsum("ab, abc->abc", pos_num_r, r_emb)
        t_emb = torch.einsum("ab, abc->abc", pos_num_r, t_emb)

        # attention weight
        att_weight = self.cal_att_weight(r_emb, t_emb, pos_num_r)
        # Equ (10)
        item_emb_nv = torch.sum(torch.mul(att_weight, t_emb), dim=1)
        item_emb = self.Item(pos_items)
        item_emb = item_emb + item_emb_nv

        user_emb = self.User(users)
        dot = torch.einsum("ac, bc->abc", user_emb, item_emb)
        pre = torch.einsum("ajk, kl->ajl", dot, self.pre_vec)
        return pre

    def infer(self, input_i, input_iu, input_hr, input_ht):
        """
        :param input_i: items
        :param input_iu:
        :param input_hr:
        :param input_ht:
        :return: total loss
        """
        # item_emb = self.Item(input_i).squeeze_()
        item_emb = self.Item(input_i)
        item_emb = torch.reshape(item_emb, [-1, self.emb_size])

        # weights
        c = self.negative_c[input_i]
        ck = self.negative_ck[input_i]

        # Dropout
        item_emb_kg = self.dropout_kg(item_emb)

        # >>> knowledge,  cal g_{hrt}
        # relations, tail entities
        r_emb = self.Relation(input_hr)
        t_emb = self.Entity(input_ht)

        # mask, for useless values 0.0, others 1.0
        pos_num_r = torch.not_equal(input_hr, self.n_relations).float()
        pos_r_emb = torch.einsum("ab, abc->abc", pos_num_r, r_emb)
        pos_t_emb = torch.einsum("ab, abc->abc", pos_num_r, t_emb)

        # Equ (6)
        pos_rt = pos_r_emb * pos_t_emb
        # pos_hrt is g_{hrt}^ in paper
        pos_hrt = torch.einsum("ac, abc->ab", item_emb_kg, pos_rt)
        pos_hrt = torch.reshape(pos_hrt, [-1, self.max_i_r])
        # <<< knowledge

        # >> CF,  cal y_{uv}
        # >>> cal items' representation q_v
        att_weight = self.cal_att_weight(pos_r_emb, pos_t_emb, pos_num_r)
        # Equ (10), e_{N_v}
        item_emb_nv = torch.sum(torch.mul(att_weight, pos_t_emb), dim=1)
        item_emb_nv_drop = self.dropout_kg(item_emb_nv)
        # Equ (10), e_v
        item_emb_cf = self.dropout_cf(item_emb)
        # Equ (10)
        item_emb_qv = item_emb_cf + item_emb_nv_drop
        # <<< cal items' representation q_v

        user_emb = self.User(input_iu)

        # mask
        pos_num_u = torch.not_equal(input_iu, self.n_users).float()
        user_emb = torch.einsum("ab, abc->abc", pos_num_u, user_emb)

        # Equ (9),  predict y_uv
        pos_iu = torch.einsum("ac, abc->abc", item_emb_qv, user_emb)
        pos_iu = torch.einsum("ajk, kl->ajl", pos_iu, self.pre_vec)

        pos_iu = torch.reshape(pos_iu, [-1, self.max_i_u])
        # << CF,  cal y_{uv}

        tot_loss = self.cal_loss(
            g_hrt=pos_hrt,
            item_batch_ev=item_emb_cf,
            item_batch_qv=item_emb_qv,
            y_uv=pos_iu,
            c=c,
            ck=ck
        )
        return tot_loss

    def forward(self, mode, **kwargs):
        if mode == "cal_att":
            return self.cal_att_weight(**kwargs)
        elif mode == "cal_loss":
            return self.infer(**kwargs)
        elif mode == "predict":
            return self.cf_score(**kwargs)

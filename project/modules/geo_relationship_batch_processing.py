import torch
from torch import nn
from utils.registry import registry
import numpy as np
from icecream import ic

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = registry.get_args("device")
        self.config = registry.get_config("model_attributes")
        self.hidden_size = self.config["hidden_size"]


class HeightWidthRelation(Base):
    def __init__(self):
        super().__init__()

    def forward(self, b1, b2):
        """
            Function:
            ---------
            This module calculate width and height relationship
            
            Parameters:
            -----------
                b1: Batch boxes 1: BS, num_boxes
                b2: Batch boxes 2: BS, num_boxes
        """
        # BS,
        x1_min, y1_min, x1_max, y1_max = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        x2_min, y2_min, x2_max, y2_max = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        # Width and Height
        w1, h1 = x1_max - x1_min, y1_max - y1_min # BS,
        w2, h2 = x2_max - x2_min, y2_max - y2_min # BS,

        w_diff = abs(w1 - w2) # BS,
        h_diff = abs(h1 - h2) # BS,

        return torch.stack([w1, h1, w2, h2, w_diff, h_diff], dim=-1).to(self.device) # BS, 5


class DistanceRelation(Base):
    def __init__(self):
        super().__init__()

    def cal2point_distance(self, p1, p2):
        """
            Function:
            ---------
            This module calculate distance of two points
            
            Parameters:
            -----------
                - p1: batch point 1: BS, 2
                - p2: batch point 2: BS, 2
        """
        p1 = p1.to(self.device).to(torch.float16)
        p2 = p2.to(self.device).to(torch.float16)
        distance = torch.norm(p1 - p2, p=2, dim=-1).unsqueeze(-1)  # L2 norm
        return distance # BS,


    def forward(self, b1, b2):
        """
            Function:
            ---------
            This module calculate min distance of two boxes
            Following [x_min, y_min, x_max, y_max] format 

            Parameters:
            -----------
                - b1: box 1: BS, 4
                - b2: box 2: BS, 4
        """
        x1_min, y1_min, x1_max, y1_max = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        x2_min, y2_min, x2_max, y2_max = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
        center_1 = torch.stack([(x1_min + x1_max) / 2, (y1_min + y1_max) / 2], dim=-1) # BS, 2
        center_2 = torch.stack([(x2_min + x2_max) / 2, (y2_min + y2_max) / 2], dim=-1) # BS, 2
        return self.cal2point_distance(center_1, center_2)


class IoURelation(Base):
    def __init__(self):
        super().__init__()
        
    def forward(self, b1, b2):
        """
            Function:
            ---------
            Calculate the IoU of two boxes
            Following [x_min, y_min, x_max, y_max] format 

            Parameters:
            -----------
                - b1: box 1: BS, 4
                - b2: box 2: BS, 4
        """
        x1_min, y1_min, x1_max, y1_max = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        x2_min, y2_min, x2_max, y2_max = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
        
        #-- Intersection [x_min, y_min, x_max, y_max]
        overlap_x = (x1_min < x2_max) & (x1_max > x2_min)
        overlap_y = (y1_min < y2_max) & (y1_max > y2_min)
        overlap = (overlap_x & overlap_y).to(self.device)

        intersection = torch.stack([
            torch.stack([x1_min, x2_min], dim=-1).max(dim=-1).values.to(self.device), # BS, 
            torch.stack([y1_min, y2_min], dim=-1).max(dim=-1).values.to(self.device), # BS,
            torch.stack([x1_max, x2_max], dim=-1).min(dim=-1).values.to(self.device), # BS,
            torch.stack([y1_max, y2_max], dim=-1).min(dim=-1).values.to(self.device), # BS,
        ], dim=-1) # BS, 4

        intersection = (intersection * overlap.unsqueeze(-1))
        
        # for id, overlap_status in enumerate(overlap):
        #     if not overlap_status:
        #         intersection[id] = torch.tensor([0, 0, 0, 0]).to(self.device)

        xi_min, yi_min, xi_max, yi_max = intersection[:, 0], intersection[:, 1], intersection[:, 2], intersection[:, 3]
            
        #-- Area
        b1_area = (x1_max - x1_min) * (y1_max - y1_min)
        b2_area = (x2_max - x2_min) * (y2_max - y2_min)
        intersection_area = (xi_max - xi_min) * (yi_max - yi_min)
        union_area = b1_area + b2_area - intersection_area

        #-- cal IoU
        iou = intersection_area / union_area
        iou_on_box1 = intersection_area / b1_area
        iou_on_box2 = intersection_area / b2_area
        
        return torch.stack([iou, iou_on_box1, iou_on_box2], dim=-1).to(self.device) # BS, 3


class AngleRelation(Base):
    def __init__(self):
        super().__init__()
        
    def forward(self, b1, b2):
        """
            Function:
            ---------
            Calculate the angle of two token via x-axis
            Following [x_min, y_min, x_max, y_max] format 

            Parameters:
            -----------
                - b1: box 1: BS, 4
                - b2: box 2: BS, 4
        """
        x1_min, y1_min, x1_max, y1_max = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        x2_min, y2_min, x2_max, y2_max = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        center_1 = torch.stack([(x1_min + x1_max) / 2, (y1_min + y1_max) / 2], dim=-1).to(self.device) # BS, 2
        center_2 = torch.stack([(x2_min + x2_max) / 2, (y2_min + y2_max) / 2], dim=-1).to(self.device) # BS, 2
        
        #-- Calculate angle
        line = center_2 - center_1
        angle = torch.atan2(line[:, 0], line[:, 1]).to(self.device)
        degree = angle * 180.0 / torch.pi

        # celsius = []
        # for d in degree:
        #     if 22.5 < d <= 67.5:
        #         celsius.append(1)
        #     elif 67.5 < d <= 112.5:
        #         celsius.append(2)
        #     elif 112.5 < d <= 157.5:
        #         celsius.append(3)
        #     elif 157.5 < d <= 202.5:
        #         celsius.append(4)
        #     elif 202.5 < d <= 247.5:
        #         celsius.append(5)
        #     elif 247.5 < d <= 292.5:
        #         celsius.append(6)
        #     elif 292.5 < d <= 337.5:
        #         celsius.append(7)
        #     elif 337.5 < d or d <= 22.5:
        #         celsius.append(8)
        celsius = ((degree + 22.5) // 45).long() % 8  # giá trị từ 0 → 7
        celsius = celsius + 1
        return celsius.unsqueeze(-1).to(torch.float16).to(self.device) # BS, 1
        


class GeoRelationship(Base):
    def __init__(self):
        super().__init__()
        self.hw_relation = HeightWidthRelation()
        self.dis_relation = DistanceRelation()
        self.iou_relation = IoURelation()
        self.angle_relation = AngleRelation()

        # Projection
        self.linear = nn.Linear(
            in_features=11,
            out_features=self.hidden_size
        )
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)

    def forward(self, target, ocr_boxes):
        expand_target = target.unsqueeze(0).expand(ocr_boxes.size(0), target.size(0)) 
        hw_relation_embed = self.hw_relation(expand_target, ocr_boxes) # num_boxes, 6
        dis_relation_embed = self.dis_relation(expand_target, ocr_boxes) # num_boxes, 1
        iou_relation_embed = self.iou_relation(expand_target, ocr_boxes) # num_boxes, 3
        angle_relation_embed = self.angle_relation(expand_target, ocr_boxes) # num_boxes, 1

        relation_embed = torch.concat([
            hw_relation_embed.to(self.device),
            dis_relation_embed.to(self.device),
            iou_relation_embed.to(self.device),
            angle_relation_embed.to(self.device),
        ], dim=-1).to(self.device)
        
        relation_embed = self.activation(
            self.layer_norm(
                self.linear(relation_embed)
            )
        )
        return relation_embed

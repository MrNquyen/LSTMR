import torch
from torch import nn
from utils.registry import registry
import numpy as np

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
                - wi: float
                    + Width of token i
                - hi: float
                    + Height of token i
                - wj: float
                    + Width of token j
                - hj: float
                    + Height of token j
        """
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        w1, h1 = x1_max - x1_min, y1_max - y1_min
        w2, h2 = x2_max - x2_min, y2_max - y2_min
        w_diff = abs(w1 - w2)
        h_diff = abs(h1 - h2)
        return torch.tensor([w1, h1, w2, h2, w_diff, h_diff]).to(self.device)


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
                - p1: point 1
                - p2: point 2
        """
        p1 = torch.tensor(p1).to(self.device)
        p2 = torch.tensor(p2).to(self.device)
        distance = torch.norm(p1 - p2, p=2)  # L2 norm
        return distance


    def forward(self, b1, b2):
        """
            Function:
            ---------
            This module calculate min distance of two boxes
            Following [x_min, y_min, x_max, y_max] format 

            Parameters:
            -----------
                - b1: box 1
                - b2: box 2
        """
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        center_1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center_2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
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
                - b1: box 1
                - b2: box 2
        """
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        
        #-- Intersection [x_min, y_min, x_max, y_max]
        overlap_x = (x1_min < x2_max) & (x1_max > x2_min)
        overlap_y = (y1_min < y2_max) & (y1_max > y2_min)
        
        if overlap_x & overlap_y:
            intersection = (
                max(x1_min, x2_min),
                max(y1_min, y2_min),
                min(x1_max, x2_max),
                min(y1_max, y2_max),
            )
        else:
            intersection = (0, 0, 0, 0)
        xi_min, yi_min, xi_max, yi_max = intersection
            
        #-- Area
        b1_area = (x1_max - x1_min) * (y1_max - y1_min)
        b2_area = (x2_max - x2_min) * (y2_max - y2_min)
        intersection_area = (xi_max - xi_min) * (yi_max - yi_min)
        union_area = b1_area + b2_area - intersection_area

        #-- cal IoU
        iou = intersection_area / union_area
        iou_on_box1 = intersection_area / b1_area
        iou_on_box2 = intersection_area / b2_area
        
        return torch.tensor([iou, iou_on_box1, iou_on_box2]).to(self.device)


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
                - b1: box 1
                - b2: box 2
        """
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        center_1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center_2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
        
        #-- Calculate angle
        line = np.abs(np.array(center_2) - np.array(center_1))
        angle = torch.atan2(line[1], line[0])
        degree = angle * 180.0 / torch.pi

        if 22.5 < degree <= 67.5:
            return 1
        elif 67.5 < degree <= 112.5:
            return 2
        elif 112.5 < degree <= 157.5:
            return 3
        elif 157.5 < degree <= 202.5:
            return 4
        elif 202.5 < degree <= 247.5:
            return 5
        elif 247.5 < degree <= 292.5:
            return 6
        elif 292.5 < degree <= 337.5:
            return 7
        elif 337.5 < degree or degree <= 22.5:
            return 8
        


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
        hw_relation_embed = [self.hw_relation(target, ocr_box) for ocr_box in ocr_boxes]
        dis_relation_embed = [self.dis_relation(target, ocr_box) for ocr_box in ocr_boxes]
        iou_relation_embed = [self.iou_relation(target, ocr_box) for ocr_box in ocr_boxes]
        angle_relation_embed = [self.angle_relation(target, ocr_box) for ocr_box in ocr_boxes]

        relation_embed = torch.concat([
            torch.stack(hw_relation_embed),
            torch.stack(dis_relation_embed),
            torch.stack(iou_relation_embed),
            torch.tensor(angle_relation_embed),
        ], dim=-1).to(self.device)
        
        relation_embed = self.activation(
            self.layer_norm(
                self.linear(relation_embed)
            )
        )
        return relation_embed

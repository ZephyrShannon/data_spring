# multi_freq_multilabel_config.py

import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, List, Optional, Union
import os

from data_alchemy.simple_stack_model import MultiFreqMultiLabelClassifier, MultiLabelTrainer


class MultiFreqMultiLabelConfig:
    """多频段多标签模型配置类"""

    DEFAULT_CONFIG = {
        'model': {
            'name': 'MultiFreqMultiLabelClassifier',
            'device': 'auto',
            'init_seed': 42,
            'deterministic_init': False,

            'low_freq': {
                'input_dim': 10,
                'seq_len': 100,
                'feature_dim': 64,

                'dnn1': {
                    'hidden_dims_list': [[32, 64]],
                    'activations': 'relu',
                    'dropouts': 0.1,
                    'use_layer_norms': True,
                    'aggregation_method': 'concat'
                },

                'cnn': {
                    'hidden_dims_list': [[128, 256]],
                    'kernel_sizes_list': [[3, 5]],
                    'dilations_list': None,
                    'strides_list': None,
                    'use_skip_connections_list': True,
                    'aggregation_method': 'concat'
                },

                'rnn': {
                    'types': 'gru',
                    'hidden_dims': 128,
                    'num_layers_list': 2,
                    'dropout_list': 0.2,
                    'aggregation_method': 'concat',
                    'use_attention_aggregation': False
                },

                'use_residual_connections': False,
                'residual_strength': 0.1
            },

            'mid_freq': {
                'input_dim': 20,
                'seq_len': 200,
                'feature_dim': 128,

                'dnn1': {
                    'hidden_dims_list': [[64, 128]],
                    'activations': 'relu',
                    'dropouts': 0.1,
                    'use_layer_norms': True,
                    'aggregation_method': 'concat'
                },

                'cnn': {
                    'hidden_dims_list': [[256, 512]],
                    'kernel_sizes_list': [[5, 7]],
                    'dilations_list': None,
                    'strides_list': None,
                    'use_skip_connections_list': True,
                    'aggregation_method': 'concat'
                },

                'rnn': {
                    'types': 'lstm',
                    'hidden_dims': 256,
                    'num_layers_list': 2,
                    'dropout_list': 0.3,
                    'aggregation_method': 'concat',
                    'use_attention_aggregation': False
                },

                'use_residual_connections': True,
                'residual_strength': 0.1,
                'use_film': True
            },

            'feature_fusion': {
                'shared_hidden_dims': [512, 256],
                'dropout': 0.3,
                'use_batch_norm': True,
                'activation': 'relu'
            },

            'classification': {
                'num_labels': 5,
                'per_label_hidden_dims': [128, 64],
                'label_dropout': 0.2,
                'use_batch_norm': True,
                'threshold': 0.5
            },

            'training': {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'num_epochs': 100,
                'early_stopping_patience': 10,
                'lr_scheduler': {
                    'enabled': True,
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                }
            },

            'evaluation': {
                'metrics': ['accuracy', 'f1_micro', 'f1_macro', 'precision', 'recall'],
                'threshold': 0.5,
                'average_methods': ['micro', 'macro', 'weighted']
            }
        }
    }

    def __init__(self, config_dict: Dict[str, Any] = None, config_path: str = None):
        """
        初始化配置

        Args:
            config_dict: 配置字典
            config_path: 配置文件路径
        """
        if config_dict is not None:
            self.config = config_dict #self._merge_configs(self.DEFAULT_CONFIG, config_dict)
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = self.DEFAULT_CONFIG.copy()

        self._validate_config()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return self._merge_configs(self.DEFAULT_CONFIG, config)

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个配置字典"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self):
        """验证配置"""
        model_cfg = self.config['model']

        # 验证必填参数
        required = [
            ('low_freq', 'input_dim'),
            ('low_freq', 'seq_len'),
            ('mid_freq', 'input_dim'),
            ('mid_freq', 'seq_len'),
            ('classification', 'num_labels')
        ]

        for section, param in required:
            if param not in model_cfg.get(section, {}):
                raise ValueError(f"必须提供 {section}.{param}")

        # 验证标签数量
        num_labels = model_cfg['classification']['num_labels']
        if num_labels <= 0:
            raise ValueError(f"标签数量必须大于0: {num_labels}")

        # 验证FiLM配置
        if model_cfg['mid_freq'].get('use_film', True):
            if 'feature_dim' not in model_cfg['low_freq']:
                raise ValueError("当使用FiLM时，必须提供low_freq.feature_dim")

    def save(self, path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"配置已保存到: {path}")

    def get_model_params(self) -> Dict[str, Any]:
        """获取模型初始化参数"""
        model_cfg = self.config['model']

        # 低频配置
        lf_cfg = model_cfg['low_freq']
        lf_params = {
            'lf_input_dim': lf_cfg['input_dim'],
            'lf_seq_len': lf_cfg['seq_len'],
            'lf_feature_dim': lf_cfg.get('feature_dim', 64),

            'lf_dnn1_hidden_dims_list': lf_cfg['dnn1']['hidden_dims_list'],
            'lf_cnn_hidden_dims_list': lf_cfg['cnn']['hidden_dims_list'],
            'lf_cnn_kernel_sizes_list': lf_cfg['cnn']['kernel_sizes_list'],
            'lf_rnn_hidden_dims': lf_cfg['rnn']['hidden_dims'],
        }

        # 中频配置
        mf_cfg = model_cfg['mid_freq']
        mf_params = {
            'mf_input_dim': mf_cfg['input_dim'],
            'mf_seq_len': mf_cfg['seq_len'],
            'mf_feature_dim': mf_cfg.get('feature_dim', 128),

            'mf_dnn1_hidden_dims_list': mf_cfg['dnn1']['hidden_dims_list'],
            'mf_cnn_hidden_dims_list': mf_cfg['cnn']['hidden_dims_list'],
            'mf_cnn_kernel_sizes_list': mf_cfg['cnn']['kernel_sizes_list'],
            'mf_rnn_hidden_dims': mf_cfg['rnn']['hidden_dims'],
            'mf_use_residual': mf_cfg.get('use_residual_connections', True),
        }

        # 特征融合配置
        fusion_cfg = model_cfg['feature_fusion']
        fusion_params = {
            'shared_hidden_dims': fusion_cfg['shared_hidden_dims'],
            'dropout': fusion_cfg.get('dropout', 0.2),
            'use_batch_norm': fusion_cfg.get('use_batch_norm', True),
        }

        # 分类配置
        cls_cfg = model_cfg['classification']
        cls_params = {
            'num_labels': cls_cfg['num_labels'],
            'per_label_hidden_dims': cls_cfg['per_label_hidden_dims'],
            'label_dropout': cls_cfg.get('label_dropout', 0.2),
        }

        # 其他配置
        other_params = {
            'device': model_cfg.get('device', 'auto'),
            'init_seed': model_cfg.get('init_seed'),
        }

        # 合并所有参数
        all_params = {**lf_params, **mf_params, **fusion_params, **cls_params, **other_params}

        # 添加FiLM需要的参数
        if mf_cfg.get('use_film', True):
            all_params['lf_feature_dim'] = lf_cfg.get('feature_dim', 64)

        return all_params

    def get_training_params(self) -> Dict[str, Any]:
        """获取训练参数"""
        train_cfg = self.config['model']['training']
        return {
            'batch_size': train_cfg['batch_size'],
            'learning_rate': train_cfg['learning_rate'],
            'weight_decay': train_cfg['weight_decay'],
            'num_epochs': train_cfg['num_epochs'],
            'early_stopping_patience': train_cfg['early_stopping_patience'],
            'lr_scheduler': train_cfg.get('lr_scheduler', {})
        }

    def get_evaluation_params(self) -> Dict[str, Any]:
        """获取评估参数"""
        eval_cfg = self.config['model']['evaluation']
        return {
            'metrics': eval_cfg['metrics'],
            'threshold': eval_cfg['threshold'],
            'average_methods': eval_cfg['average_methods']
        }

    def __getitem__(self, key):
        """支持字典式访问"""
        return self.config['model'].get(key, {})

    def __repr__(self):
        """友好的字符串表示"""
        import json
        return json.dumps(self.config, indent=2, ensure_ascii=False)


# ==================== 模型工厂函数 ====================

def create_model_from_config(config_path: str = None,
                             config_dict: Dict[str, Any] = None) -> MultiFreqMultiLabelClassifier:
    """
    从配置创建模型

    Args:
        config_path: 配置文件路径
        config_dict: 配置字典

    Returns:
        MultiFreqMultiLabelClassifier实例
    """
    # 加载配置
    config = MultiFreqMultiLabelConfig(config_dict=config_dict, config_path=config_path)

    # 获取模型参数
    model_params = config.get_model_params()

    # 设置全局种子（如果需要）
    init_seed = config.config['model'].get('init_seed')
    deterministic = config.config['model'].get('deterministic_init', False)

    if init_seed is not None:
        _set_global_seed(init_seed, deterministic)
        print(f"使用种子 {init_seed} 初始化模型")

    # 创建模型
    model = MultiFreqMultiLabelClassifier(**model_params)

    # 打印配置信息
    _print_config_summary(config)

    return model


def _set_global_seed(seed: int, deterministic: bool = False):
    """设置全局随机种子"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _print_config_summary(config: MultiFreqMultiLabelConfig):
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("模型配置摘要")
    print("=" * 70)

    model_cfg = config.config['model']

    # 基本信息
    print(f"模型名称: {model_cfg.get('name', 'MultiFreqMultiLabelClassifier')}")
    print(f"设备: {model_cfg.get('device', 'auto')}")

    # 低频配置
    lf = model_cfg['low_freq']
    print(f"\n低频配置:")
    print(f"  输入: ({lf['seq_len']}, {lf['input_dim']})")
    print(f"  特征维度: {lf.get('feature_dim', 64)}")
    print(f"  DNN1: {lf['dnn1']['hidden_dims_list']}")
    print(f"  CNN: {lf['cnn']['hidden_dims_list']} (kernel: {lf['cnn']['kernel_sizes_list']})")
    print(f"  RNN: {lf['rnn']['hidden_dims']} ({lf['rnn']['types']})")

    # 中频配置
    mf = model_cfg['mid_freq']
    print(f"\n中频配置:")
    print(f"  输入: ({mf['seq_len']}, {mf['input_dim']})")
    print(f"  特征维度: {mf.get('feature_dim', 128)}")
    print(f"  DNN1: {mf['dnn1']['hidden_dims_list']}")
    print(f"  CNN: {mf['cnn']['hidden_dims_list']} (kernel: {mf['cnn']['kernel_sizes_list']})")
    print(f"  RNN: {mf['rnn']['hidden_dims']} ({mf['rnn']['types']})")
    print(f"  使用FiLM: {mf.get('use_film', True)}")
    print(f"  使用残差连接: {mf.get('use_residual_connections', True)}")

    # 分类配置
    cls = model_cfg['classification']
    print(f"\n分类配置:")
    print(f"  标签数量: {cls['num_labels']}")
    print(f"  每个标签分类器: {cls['per_label_hidden_dims']}")

    # 特征融合
    fusion = model_cfg['feature_fusion']
    print(f"\n特征融合配置:")
    print(f"  共享层: {fusion['shared_hidden_dims']}")
    print(f"  Dropout: {fusion.get('dropout', 0.3)}")
    print("=" * 70 + "\n")


# ==================== 训练配置器 ====================

class TrainingConfigurator:
    """训练配置器"""

    def __init__(self, config: MultiFreqMultiLabelConfig):
        self.config = config
        self.model = None
        self.trainer = None

    def create_model_and_trainer(self) -> tuple:
        """创建模型和训练器"""
        # 创建模型
        self.model = create_model_from_config(config_dict=self.config.config)

        # 创建训练器
        self.trainer = MultiLabelTrainer(self.model)

        # 设置优化器
        train_params = self.config.get_training_params()
        self.trainer.setup_optimizer(
            lr=train_params['learning_rate'],
            weight_decay=train_params['weight_decay']
        )

        return self.model, self.trainer

    def create_lr_scheduler(self, optimizer):
        """创建学习率调度器"""
        lr_cfg = self.config.config['model']['training'].get('lr_scheduler', {})

        if not lr_cfg.get('enabled', False):
            return None

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_cfg.get('factor', 0.5),
            patience=lr_cfg.get('patience', 5),
            min_lr=lr_cfg.get('min_lr', 1e-6),
            verbose=True
        )

        return scheduler

    def get_early_stopping_config(self) -> Dict[str, Any]:
        """获取早停配置"""
        train_cfg = self.config.config['model']['training']
        return {
            'patience': train_cfg['early_stopping_patience'],
            'min_delta': 1e-5,
            'verbose': True
        }


# ==================== 示例配置文件生成 ====================

def generate_example_configs():
    """生成示例配置文件"""

    # 1. 小型配置
    small_config = {
        'model': {
            'name': 'SmallMultiFreqModel',
            'low_freq': {
                'input_dim': 5,
                'seq_len': 50,
                'feature_dim': 32,
                'dnn1': {'hidden_dims_list': [[16, 32]]},
                'cnn': {'hidden_dims_list': [[64, 128]], 'kernel_sizes_list': [[3, 5]]},
                'rnn': {'hidden_dims': 64, 'types': 'gru'}
            },
            'mid_freq': {
                'input_dim': 10,
                'seq_len': 100,
                'feature_dim': 64,
                'dnn1': {'hidden_dims_list': [[32, 64]]},
                'cnn': {'hidden_dims_list': [[128, 256]], 'kernel_sizes_list': [[5, 7]]},
                'rnn': {'hidden_dims': 128, 'types': 'gru'}
            },
            'classification': {
                'num_labels': 3,
                'per_label_hidden_dims': [64, 32]
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-3,
                'num_epochs': 50
            }
        }
    }

    # 2. 大型配置
    large_config = {
        'model': {
            'name': 'LargeMultiFreqModel',
            'low_freq': {
                'input_dim': 20,
                'seq_len': 200,
                'feature_dim': 128,
                'dnn1': {'hidden_dims_list': [[64, 128, 256]]},
                'cnn': {'hidden_dims_list': [[256, 512, 1024]], 'kernel_sizes_list': [[3, 5, 7]]},
                'rnn': {'hidden_dims': 256, 'types': 'lstm', 'num_layers_list': 3}
            },
            'mid_freq': {
                'input_dim': 40,
                'seq_len': 400,
                'feature_dim': 256,
                'dnn1': {'hidden_dims_list': [[128, 256, 512]]},
                'cnn': {'hidden_dims_list': [[512, 1024, 2048]], 'kernel_sizes_list': [[5, 7, 9]]},
                'rnn': {'hidden_dims': 512, 'types': 'lstm', 'num_layers_list': 3}
            },
            'feature_fusion': {
                'shared_hidden_dims': [1024, 512, 256]
            },
            'classification': {
                'num_labels': 10,
                'per_label_hidden_dims': [256, 128, 64]
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 5e-4,
                'num_epochs': 200,
                'early_stopping_patience': 20
            }
        }
    }

    # 3. 特殊配置：不使用FiLM
    no_film_config = {
        'model': {
            'mid_freq': {
                'use_film': False
            }
        }
    }

    # 保存配置文件
    configs = {
        'config_small.yaml': small_config,
        'config_large.yaml': large_config,
        'config_no_film.yaml': no_film_config
    }

    for filename, config in configs.items():
        # 创建完整配置
        full_config = MultiFreqMultiLabelConfig(config_dict=config)
        full_config.save(filename)
        print(f"已生成配置文件: {filename}")

    # 生成默认配置文件
    default_config = MultiFreqMultiLabelConfig()
    default_config.save('config_default.yaml')
    print(f"已生成默认配置文件: config_default.yaml")


# ==================== 配置验证工具 ====================

def validate_config_file(config_path: str) -> bool:
    """验证配置文件是否有效"""
    try:
        config = MultiFreqMultiLabelConfig(config_path=config_path)
        print(f"✅ 配置文件验证通过: {config_path}")
        print(f"   标签数量: {config['classification']['num_labels']}")
        print(f"   低频输入: ({config['low_freq']['seq_len']}, {config['low_freq']['input_dim']})")
        print(f"   中频输入: ({config['mid_freq']['seq_len']}, {config['mid_freq']['input_dim']})")
        return True
    except Exception as e:
        print(f"❌ 配置文件验证失败: {config_path}")
        print(f"   错误: {str(e)}")
        return False


# ==================== 使用示例 ====================

def config_example():
    """配置使用示例"""
    print("多频段多标签模型配置示例")
    print("=" * 60)

    # 1. 生成示例配置文件
    print("\n1. 生成示例配置文件...")
    generate_example_configs()

    # 2. 验证配置文件
    print("\n2. 验证配置文件...")
    validate_config_file('config_default.yaml')

    # 3. 从配置文件创建模型
    print("\n3. 从配置文件创建模型...")
    try:
        model = create_model_from_config('config_default.yaml')

        # 测试模型
        batch_size = 4
        lf_input = torch.randn(batch_size, 100, 10)
        mf_input = torch.randn(batch_size, 200, 20)

        with torch.no_grad():
            output = model(lf_input, mf_input)
            print(f"模型测试通过!")
            print(f"输入形状: 低频{lf_input.shape}, 中频{mf_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

        # 4. 创建训练器
        print("\n4. 创建训练配置器...")
        config = MultiFreqMultiLabelConfig(config_path='config_default.yaml')
        configurator = TrainingConfigurator(config)
        model, trainer = configurator.create_model_and_trainer()

        print(f"训练器创建成功!")
        print(f"学习率: {config['training']['learning_rate']}")
        print(f"批次大小: {config['training']['batch_size']}")

    except Exception as e:
        print(f"创建模型失败: {str(e)}")


# ==================== 命令行工具 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='多频段多标签模型配置工具')
    parser.add_argument('--generate', action='store_true', help='生成示例配置文件')
    parser.add_argument('--validate', type=str, help='验证配置文件')
    parser.add_argument('--create', type=str, help='从配置文件创建模型')
    parser.add_argument('--example', action='store_true', help='运行完整示例')

    args = parser.parse_args()

    if args.generate:
        generate_example_configs()
    elif args.validate:
        validate_config_file(args.validate)
    elif args.create:
        model = create_model_from_config(args.create)
        print(f"从 {args.create} 创建模型成功!")
        print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    elif args.example:
        config_example()
    else:
        print("使用 --help 查看可用命令")
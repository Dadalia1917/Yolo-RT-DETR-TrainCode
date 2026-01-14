"""
GPU 检测工具 - 用于确认 CUDA 设备索引与物理 GPU 的对应关系
运行此脚本来查看系统中所有 GPU 的信息.
"""

import os

import torch


def print_gpu_info():
    print("=" * 70)
    print("GPU 设备检测工具")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用!")
        return

    gpu_count = torch.cuda.device_count()
    print(f"\n检测到 {gpu_count} 个 CUDA 设备:\n")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        # 尝试获取 PCI 总线信息
        try:
            pci_bus_id = torch.cuda.device_get_pci_bus_id_string(i)
        except:
            pci_bus_id = "未知"

        print(f"  CUDA 设备 {i}:")
        print(f"    名称: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    PCI 总线 ID: {pci_bus_id}")

        # 进行简单的 GPU 性能测试来区分
        torch.cuda.set_device(i)
        # 测试一下
        test_tensor = torch.randn(1000, 1000, device=f"cuda:{i}")
        del test_tensor
        torch.cuda.empty_cache()
        print()

    print("=" * 70)
    print("如何区分两块卡:")
    print("  1. 使用 nvidia-smi 查看完整的 PCI 总线地址")
    print("  2. 或者启动训练后，观察任务管理器中哪块卡的 GPU 利用率上升")
    print("=" * 70)
    print("\n使用说明:")
    print("  在训练脚本中设置 device='0' 使用第一个设备")
    print("  在训练脚本中设置 device='1' 使用第二个设备")
    print("=" * 70)

    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        print(f"\n注意: CUDA_VISIBLE_DEVICES = {cuda_visible}")
        print("这可能会影响 GPU 索引的映射!")


if __name__ == "__main__":
    print_gpu_info()

    # 额外测试：在指定 GPU 上运行简单任务
    print("\n" + "=" * 70)
    print("GPU 使用测试")
    print("=" * 70)

    for i in range(torch.cuda.device_count()):
        print(f"\n正在测试 CUDA 设备 {i}...")
        device = torch.device(f"cuda:{i}")

        # 运行一个简单的计算
        a = torch.randn(5000, 5000, device=device)
        b = torch.randn(5000, 5000, device=device)
        c = torch.matmul(a, b)

        print(f"  设备 {i} 测试完成 - 请观察任务管理器中哪个 GPU 的利用率上升")
        print("  (等待2秒...)")

        import time

        time.sleep(2)

        del a, b, c
        torch.cuda.empty_cache()

    print("\n测试完成!")

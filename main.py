import torch

# from bcos.training import trainer
# from bcos.experiments.utils import get_configs_and_model_factory

# def main():
#     print("Hello from woofsandmeows!")

# def train_bcos():
#     ## We have to see what is most convenient, calling the function like this directly or using the CLI. 
#     trainer.run_training()

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.hip)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

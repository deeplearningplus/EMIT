from prettytable import PrettyTable
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/22?page=2
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters (Mb)"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel() / 1e6
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

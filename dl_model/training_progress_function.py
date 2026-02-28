def train_progress(epoch: int, epochs: int, loss_function, show) -> str:
    """Training Progress Function:
    """
    if show: 
        if (epoch % 10 == 0):
            print(f"⏳ Training progress | {(epoch / epochs) * 100}% | Loss = {loss_function.item()}") 
        if (epoch == epochs - 1):
            print(f"✅ Model Trained: Training Progress | 100% | Loss = {loss_function.item()}")
    else: 
        pass
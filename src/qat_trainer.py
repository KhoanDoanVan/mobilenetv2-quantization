import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class QATTrainer:
    """
    STEP 4: Quantization-Aware Training

    Fine-tune model with FakeQuantization enabled
    """

    def __init__(
            self,
            model,
            config
    ):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.QAT_EPOCHS
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def enable_quantization(self):
        """
        Enable FakeQuantization in all layers
        """
        for module in self.model.modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quantization.enabled = True
            if hasattr(module, 'activation_fake_quant'):
                module.activation_fake_quant.enabled = True


    def disable_quantization(self):
        """
        Disable FakeQuantization (warmup phase)
        """
        for module in self.model.modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.enabled = False
            if hasattr(module, 'activation_fake_quant'):
                module.activation_fake_quant.enabled = False


    def train_epoch(self, dataloader, epoch):
        """
        Train an epoch
        """

        self.model.train()

        # Warmup: disable quantization at few first epochs
        if epoch < self.config.QUANT_DELAY:
            self.disable_quantization()
        else:
            self.enable_quantization()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.QAT_EPOCHS}')

        for batch_idx, (inputs, targets) in enumerate(pbar):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        # Step scheduler
        self.scheduler.step()

        return total_loss / len(dataloader), 100. * correct / total
    

    def validate(self, dataloader):
        """
        Validate model
        """

        self.model.eval()
        self.enable_quantization()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(dataloader), 100. * correct / total
    

    def train(self, train_loader, val_loader):
        """
        Full QAT training loop
        """

        best_acc = 0

        for epoch in range(self.config.QAT_EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.QAT_EPOCHS}")
            print(f"{'='*60}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            print(f"\nTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc
                }
                torch.save(checkpoint, f'{self.config.QAT_MODEL_DIR}/best_qat_model.pth')
                print(f"✓ Saved best model with Val Acc: {best_acc:.2f}%")

        return best_acc
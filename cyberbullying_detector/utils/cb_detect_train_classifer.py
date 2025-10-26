import torch
from cyberbullying_detector.utils.cb_detect_loss_funcs import calc_loss_batch, calc_accuracy_loader
from cyberbullying_detector.utils.cb_detect_evaluate import evaluate_model
from global_utils import log_message

def train_classifier(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, scheduler=None, grad_clip=1.0, log_freq=50
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            # Log progress for immediate feedback
            if global_step % log_freq == 0:
                log_message(f"Epoch {epoch+1}, Step {global_step:06d}: Loss {loss.item():.4f}, Examples seen: {examples_seen}")

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                log_message(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        log_message(f"Training accuracy: {train_accuracy * 100:.2f}%")
        log_message(f"Validation accuracy: {val_accuracy * 100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            log_message(f"Learning rate: {current_lr:.2e}")

    return train_losses, val_losses, train_accs, val_accs, examples_seen

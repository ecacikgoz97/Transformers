import torch
from utils import translate_sentence, save_checkpoint

def training(epochs, model, optimizer, criterion, scheduler, train_loader, sentence, german, english, save_model, device, writer):
    step=0
    for epoch in range(epochs):
        print(f"[Epoch {epoch} / {epochs}]")

        if save_model:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)

        model.eval()
        translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)

        print(f"Translated example sentence: \n {translated_sentence}")
        model.train()
        losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

import torch

def train_nf(tr_dataloader, nf, opt, num_epochs, verbose_num_iters=100, data_type="2d"):
    nf.train()
    nf.cuda()
    loss_trace = []

    iter_i = 0

    for epoch_i in range(num_epochs):
        print(f"Epoch {epoch_i + 1}")
        for batch in tr_dataloader:
            x, _ = batch
            x.cuda()
            #x = x.to(device)

            if data_type == "mnist":
                x = x.view(x.shape[0], -1)
                # деквантизация
                x = x + 0.05 * torch.randn_like(x)

            # делаем шаг обучения
            opt.zero_grad()
            loss = -nf.log_prob(x)
            loss.backward()
            opt.step()
            loss_trace.append((iter_i, loss.item()))

            iter_i += 1

            # if iter_i % verbose_num_iters == 0:
            #     clear_output(wait=True)
            #     plt.figure(figsize=(10, 5))
            #
            #     plt.subplot(1, 2, 1)
            #     plt.xlabel("Iteration")
            #     plt.ylabel("Normalizing flow loss")
            #     plt.plot([p[0] for p in loss_trace], [p[1] for p in loss_trace])
            #     plt.show()
            #     nf.train()

    nf.eval()
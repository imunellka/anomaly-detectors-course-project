# Anomaly detectors
– Предсказание промышленных кибер атак с использованием нормализующих потоков. Временные ряды.
  
---

### **Аннотация:**
В данной работе исследуется возможность предсказания кибератак на промышленные системы и процессы при помощи нейросетевых подходов. Исследование охватывает область применения вариационных и обычных автокодировщиков, а также моделей трансформеров. Дополнительно будет рассмотрен вариант использования нормализующих потоков для улучшения детекции аномальных объектов. Данное исследование направлено на помощь сотрудникам по кибербезопасности и инженерам в оперативном выявлении различных видов кибератак.

### **Были рассмотрены:**
- SOTA подходы в области детекции аномалий
- Возможности и ограничения в сфере ICS
- Особенности детекции аномалий во временных рядах


### **Используемые модели:**
1) [AutoEncoder](https://homes.cs.aau.dk/~byang/papers/IJCAI2019.pdf)
2) [VAE](https://neerc.ifmo.ru/wiki/index.php?title=Вариационный_автокодировщик)
3) [Transformers](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
4) [RealNVP](https://yuanzhi-zhu.github.io/2022/06/21/Real-NVP-Intro/)

### **Датасет:**
Можно найти по этой [ссылке](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)

### **Использование:**
---

Библиотека представляет из себя три пакета:

`utils`:
- `backprop.py` – функция имплементирующая механизм обратного распространения ошибки, используется в файле training.py. На вход подается (epoch, model, data, dataO, optimizer, scheduler, training = True)
- `create_window.py` – функция преобразования датасета из временных рядов в датасет окон. На вход подается data( torch.Tensor) и разме окна.
- `positional_encoding.py` - внутренняя функция, имплементирует позиционное кодирование, принимает вектор
- `training.py` – Вспомогательная функция для обучения. На вход подается - model, trainD, test_final, label, num_epochs, optimizer, scheduler
- `training_flow.py` – Вспомогатлеьная функция для обучения потока. На вход подается dataloader, model, optimizer, num_epochs

`models`:
- `TransformerBasic.py` - Базовая модель. На вход подается  количество фичей.
- `TransformerAutoencoder.py` - Автоэнкодер основанный на трансформер-блоках. На вход подается  количество фичей, lr, window_size.
- `TransformerAutoencoderModified.py` - Модифицированный автоэнкодер основанный на трансформер-блоках. На вход подается  количество фичей, lr, window_size и batch_size.
- `VAE_trasformer.py` – Вариационный автоэнкодер, основанный на трансформер-блоках. На вход подается  количество фичей, lr, window_size и размерность скрытого пространства.
- `real_nvp.py` - Нормализующий поток Real-Nvp. На вход подается количество фичей и маска.


`evaluate`:
- `evaluation.py` - Файл с реализованным подсчетом метрик.



### **Examples:**
``` 
# преобразуем данные в окна
trainD_prom,testD_prom = split_train_test_windows(torch_data, 15)
```

``` 
# создаем модель TransformerAutoencoderModified, импорт модели делается стандартно
model = TransformerAutoEncoderModified(7,0.001,50,128) # количество фичей равно 7, lr = 0.001, размер окна = 50, размер батча 128
```

``` 
# создаем модель VAE_TransformerBasic, импорт модели делается стандартно
model = VAE_TransformerBasic(7, 0.001,30,7) # количество фичей равно 7, lr = 0.001, размер окна = 30, размер латентного пространства = 7
```

``` 
# создаем модель RealNVP, импорт модели делается стандартно

layers = []
for i in range(16):
    mask = ((torch.arange(4) + i) % 2).cuda()
    layers.append(RealNVP(var_size=4, mask=mask))

nf = NormalizingFlow(layers=layers, prior=torch.distributions.MultivariateNormal(torch.zeros(4).cuda(), torch.eye(4).cuda()))

train_nf(trainloader, nf, optimizer, num_epochs=50)
```

### **Результаты:**
Обзор на все результаты представлен в прикрепленном отчете




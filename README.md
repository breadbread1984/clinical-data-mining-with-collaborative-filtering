# clinical-data-mining-with-collaborative-filtering

this project use neural collaborative filtering to do clinic data miniing

## prepare pretrain data

create dataset with the following command

```python
python3 create_dataset.py
```

## pretrain model

pretrain models with the following command

```python
python3 pretrain.py (GMF|MLP) (ml-1m|pinterest)
```

copy the generated model.h5 to gmf.h5 and mlp.h5 manually.

## train NeuMF model

find the optimal hyper parameter first with the following command

```python
python3 train.py
```

## quantilize medical clinic data

quantilize with the following command

```python
python3 quantilize.py
```

## train NeuMF on the clinic data

train with 

```python
python3 train_clinic.py
```



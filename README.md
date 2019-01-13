# vrnn2018

## [Chess games database](https://www.ficsgames.org/download.html)

## How to play

```shell
$ source env/bin/activate
(env) $ pip3 install -r requirements.txt
(env) $ pip3 install -e .
(env) $ python3 gen/generate_games.py --randomgames 10 --bookgames 10
(env) $ python3 models/dual_v1.py --epochs 50 --gpus "2,3"
```

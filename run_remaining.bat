@echo off
echo Running remaining hyperparameter configurations...

echo.
echo ========================================
echo Configuration 3: Larger batch size
echo ========================================
python scripts/02_train_t5_tf.py --lr 3e-4 --batch_size 32 --epochs 5 --label_smoothing 0.1 --warmup_ratio 0.05 --dropout 0.1 --weight_decay 0.01 --track --notes "Larger batch size" --save_dir models/t5-small-finance-run3

echo.
echo ========================================
echo Configuration 4: More epochs + regularization
echo ========================================
python scripts/02_train_t5_tf.py --lr 3e-4 --batch_size 16 --epochs 8 --label_smoothing 0.2 --warmup_ratio 0.1 --dropout 0.2 --weight_decay 0.01 --track --notes "More epochs + strong regularization" --save_dir models/t5-small-finance-run4

echo.
echo ========================================
echo Configuration 5: Lower LR, longer training
echo ========================================
python scripts/02_train_t5_tf.py --lr 1e-4 --batch_size 16 --epochs 8 --label_smoothing 0.1 --warmup_ratio 0.1 --dropout 0.1 --weight_decay 0.01 --track --notes "Lower LR, longer training" --save_dir models/t5-small-finance-run5

echo.
echo ========================================
echo Configuration 6: Aggressive training
echo ========================================
python scripts/02_train_t5_tf.py --lr 5e-4 --batch_size 32 --epochs 8 --label_smoothing 0.1 --warmup_ratio 0.05 --dropout 0.15 --weight_decay 0.01 --track --notes "Aggressive training" --save_dir models/t5-small-finance-run6

echo.
echo ========================================
echo Configuration 7: Conservative approach
echo ========================================
python scripts/02_train_t5_tf.py --lr 1e-4 --batch_size 8 --epochs 5 --label_smoothing 0.05 --warmup_ratio 0.1 --dropout 0.1 --weight_decay 0.0 --track --notes "Conservative approach" --save_dir models/t5-small-finance-run7

echo.
echo ========================================
echo All configurations complete!
echo ========================================
python scripts/02_train_t5_tf.py --summary

pause
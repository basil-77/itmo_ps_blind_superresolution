# Blind image super-resulution using self-supervised learning
Проект в рамках дисциплины "Проектный семинар" AI Talent Hub ИТМО.

## Поставновка задачи
Blind image super-resolution – восстановления изображения в высоком разрешении $I_HR$ из изображения в низком разрешении $I_LR$ при условии неизвестного метода деградации $F_D$. Изображения $I_HR$ на этапе обучения недоступны:  
$I_LR = F_D(I_HR, s)$, где  
$F_D$ - degradation function,  
$I_HR$ - source HR image,  
$s$ - scale factor


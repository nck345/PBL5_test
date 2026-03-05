@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHON_EXE=C:\Users\khiem_ssxdm6v\AppData\Local\Python\bin\python.exe"

:: Kiểm tra xem có lệnh nào được truyền vào không
if "%~1"=="" (
    echo [Loi] Vui long nhap lenh can chay!
    echo Cac lenh ho tro:
    echo   .\run.bat train           (Huấn luyện mô hình^)
    echo   .\run.bat test            (Đánh giá mô hình^)
    echo   .\run.bat predict "PATH"  (Dự đoán file/thư mục^)
    exit /b 1
)

:: Xử lý Menu Lệnh
set "COMMAND=%~1"

if /i "%COMMAND%"=="train" (
    echo [Thong bao] Dang khoi dong qua trinh Train...
    "%PYTHON_EXE%" train.py %2 %3 %4 %5 %6 %7 %8 %9
    
) else if /i "%COMMAND%"=="test" (
    echo [Thong bao] Dang khoi dong qua trinh Test...
    "%PYTHON_EXE%" test.py %2 %3 %4 %5 %6 %7 %8 %9
    
) else if /i "%COMMAND%"=="predict" (
    if "%~2"=="" (
        echo [Loi] Thieu duong dan! Vui long nhap: .\run.bat predict "DUONG_DAN" [--stream] [TEN_FILE_BAO_CAO]
        exit /b 1
    )
    
    echo [Thong bao] Dang chay du doan...
    "%PYTHON_EXE%" predict.py --input "%~2" %3 %4 %5 %6 %7 %8 %9
    
) else (
    echo [Thong bao] Chay lenh tuy chinh...
    "%PYTHON_EXE%" %*
)

endlocal

:: ======================================================================
:: DANH SACH CAC MAU LENH PHO BIEN
:: 
:: 1. Quet 1 file va tu dong luu bao cao mac dinh:
:: .\run.bat predict "dataset/Raw Data/SCH/SCH_acc_10_2.txt"
::
:: 2. Quet 1 file, luu bao cao voi ten file chi dinh:
:: .\run.bat predict "dataset/Raw Data/SCH/SCH_acc_10_2.txt" bao_cao_cua_toi.txt
::
:: 3. Quet Streaming 1 file, tu dong luu nhat ky mac dinh (json):
:: .\run.bat predict "dataset/Raw Data/SCH/SCH_acc_10_2.txt" --stream
::
:: 4. Quet Streaming 1 file, luu nhat ky thanh ten json chi dinh:
:: .\run.bat predict "dataset/Raw Data/SCH/SCH_acc_10_2.txt" --stream nhat_ky_trinh_dien.json
::
:: 5. Quet TOAN BO THU MUC va tu dong luu bao cao mac dinh:
:: .\run.bat predict "dataset/Raw Data/SCH"
::
:: 6. Quet TOAN BO THU MUC, luu bao cao thanh ten txt chi dinh:
:: .\run.bat predict "dataset/Raw Data/SCH" bao_cao_tong_ket.txt
:: ======================================================================

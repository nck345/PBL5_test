import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Patch audit_activity_file_counts
    old_audit = '''def audit_activity_file_counts(raw_data_dir: str, labels: list,
                               sensor_type: str = "acc") -> tuple:
    """
    Dem so file theo tung activity label trong config.

    Returns:
        (counts, missing_labels)
    """
    counts = {}
    missing_labels = []

    for label in labels:
        activity_code = str(label).upper()
        activity_path = os.path.join(raw_data_dir, activity_code)
        pattern = os.path.join(activity_path, f"{activity_code}_{sensor_type}_*.txt")
        count = len(glob.glob(pattern)) if os.path.isdir(activity_path) else 0
        counts[activity_code] = count
        if count == 0:
            missing_labels.append(activity_code)

    return counts, missing_labels'''

    new_audit = '''def audit_activity_file_counts(raw_data_dir: str, labels: list,
                               sensor_type: str = "acc") -> tuple:
    """
    Dem so file theo tung activity label trong config.

    Returns:
        (counts, missing_labels)
    """
    counts = {}
    missing_labels = []

    for label in labels:
        activity_code = str(label).upper()
        activity_path = os.path.join(raw_data_dir, activity_code)
        pattern = os.path.join(activity_path, "*.csv")
        count = len(glob.glob(pattern)) if os.path.isdir(activity_path) else 0
        counts[activity_code] = count
        if count == 0:
            missing_labels.append(activity_code)

    return counts, missing_labels'''

    if old_audit in content:
        content = content.replace(old_audit, new_audit)
        print("Patched audit_activity_file_counts")
    else:
        print("Could not find old_audit")

    # 2. Patch load_mobiact_data block
    old_load = '''        # Lấy danh sách file của cảm biến đầu tiên để làm gốc
        first_sensor = sensors[0]
        pattern = os.path.join(activity_path, f"{activity_code}_{first_sensor}_*.txt")
        first_sensor_files = sorted(glob.glob(pattern))

        if not first_sensor_files:
            if verbose:
                print(f"  Không tìm thấy file {first_sensor} cho {activity_code}")
            continue

        for fpath in first_sensor_files:
            # Phân tích Subject và Trial từ filename
            # Format: Activity_Sensor_Subject_Trial.txt
            basename = os.path.basename(fpath)
            parts = basename.replace('.txt', '').split('_')
            
            # parts: [Activity, Sensor, Subject, Trial]
            if len(parts) < 4:
                continue
            
            subject_id_str = parts[2]
            trial_id_str = parts[3]
            
            # Tìm các file cảm biến khác tương ứng
            res_first = parse_mobiact_file(fpath)
            acc_data = res_first.get('data', np.array([]))
            min_len = acc_data.shape[0]
            if min_len < 10:
                continue
                
            combined_data = None
            if len(sensors) > 1 and "gyro" in sensors:
                gyro_fpath = os.path.join(activity_path, f"{activity_code}_gyro_{subject_id_str}_{trial_id_str}.txt")
                has_gyro = False
                if os.path.exists(gyro_fpath):
                    res_gyro = parse_mobiact_file(gyro_fpath)
                    gyro_data = res_gyro.get('data', np.array([]))
                    if gyro_data.shape[0] >= 10:
                        has_gyro = True
                        
                if has_gyro:
                    min_len = min(min_len, gyro_data.shape[0])
                    # Flag = 1 báo hiệu là có Gyro xịn
                    flag_col = np.ones((min_len, 1))
                    combined_data = np.hstack([acc_data[:min_len, :], gyro_data[:min_len, :], flag_col])
                else:
                    skipped_due_to_missing_sensor += 1
                    # Thiết kế đắp Zero-Pad (Bù số 0) thay vì ném bỏ toàn bộ file ghi
                    gyro_pad = np.zeros((min_len, 3))
                    # Flag = 0 ngầm hiểu là tắt nhánh Gyro trong MultiBranchLSTM
                    flag_col = np.zeros((min_len, 1))
                    combined_data = np.hstack([acc_data[:min_len, :], gyro_pad, flag_col])
            else:
                # Nếu chỉ chạy hệ gốc 'acc'
                combined_data = acc_data[:min_len, :]

            all_segments.append(combined_data)
            all_labels.append(label)
            all_subjects.append(int(subject_id_str))'''

    new_load = '''        # Lấy danh sách file csv (MobiAct Annotated Data có định dạng csv)
        pattern = os.path.join(activity_path, "*.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            if verbose:
                print(f"  Không tìm thấy file csv cho {activity_code}")
            continue

        for fpath in csv_files:
            # Phân tích Subject và Trial từ filename
            # Format: BSC_10_1_annotated.csv -> parts: [Activity, Subject, Trial, "annotated"]
            basename = os.path.basename(fpath)
            parts = basename.replace('.csv', '').split('_')
            
            if len(parts) < 3:
                continue
            
            subject_id_str = parts[1]
            trial_id_str = parts[2]
            
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
                
            if 'acc_x' in df.columns:
                acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
            else:
                continue

            min_len = acc_data.shape[0]
            if min_len < 10:
                continue
                
            combined_data = None
            if len(sensors) > 1 and "gyro" in sensors:
                has_gyro = False
                if 'gyro_x' in df.columns and not df['gyro_x'].isnull().all():
                    gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
                    has_gyro = True
                        
                if has_gyro:
                    # Flag = 1 báo hiệu là có Gyro xịn
                    flag_col = np.ones((min_len, 1))
                    combined_data = np.hstack([acc_data, gyro_data, flag_col])
                else:
                    skipped_due_to_missing_sensor += 1
                    # Thiết kế đắp Zero-Pad (Bù số 0) thay vì ném bỏ toàn bộ file ghi
                    gyro_pad = np.zeros((min_len, 3))
                    # Flag = 0 ngầm hiểu là tắt nhánh Gyro trong MultiBranchLSTM
                    flag_col = np.zeros((min_len, 1))
                    combined_data = np.hstack([acc_data, gyro_pad, flag_col])
            else:
                # Nếu chỉ chạy hệ gốc 'acc'
                combined_data = acc_data

            all_segments.append(combined_data)
            all_labels.append(label)
            all_subjects.append(int(subject_id_str))'''

    if old_load in content:
        content = content.replace(old_load, new_load)
        print("Patched load_mobiact_data")
    else:
        print("Could not find old_load")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == "__main__":
    patch_file("src/dataset.py")

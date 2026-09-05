"""
ICD-9 code -> broad clinical category mapping.
This mirrors the map_icd9() function from the notebook exactly, so a
diagnosis code entered in the app gets categorized the same way it was
during training. If you changed this function in your notebook, update
it here too -- these must stay in sync.
"""
import pandas as pd


def map_icd9(val):
    if pd.isna(val) or val == '?' or val == '':
        return 'Missing'
    try:
        num_val = float(val.split('V')[0].split('E')[0]) if 'V' not in str(val) and 'E' not in str(val) else -1
        if 390 <= num_val <= 459 or num_val == 785:
            return 'Circulatory'
        elif 460 <= num_val <= 519 or num_val == 786:
            return 'Respiratory'
        elif 520 <= num_val <= 579 or num_val == 787:
            return 'Digestive'
        elif str(val).startswith('250'):
            return 'Diabetes'
        elif 800 <= num_val <= 999:
            return 'Injury'
        elif 710 <= num_val <= 739:
            return 'Musculoskeletal'
        elif 580 <= num_val <= 629 or num_val == 788:
            return 'Genitourinary'
        elif 140 <= num_val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except Exception:
        return 'Other'

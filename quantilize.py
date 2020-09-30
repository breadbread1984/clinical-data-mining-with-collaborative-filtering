#!/usr/bin/python3

import xlrd;
import pickle;

labels_collection = {
  'Sex': {'Blank(s)': 0},
  'Primary Site - labeled': {'Blank(s)': 0},
  'Grade': {'Blank(s)': 0},
  'Laterality': {'Blank(s)': 0},
  'Summary stage 2000 (1998+)': {'Blank(s)': 0},
  'Derived AJCC Stage Group, 7th ed (2010-2015)': {'Blank(s)': 0},
  'Derived AJCC T, 7th ed (2010-2015)': {'Blank(s)': 0},
  'Derived AJCC N, 7th ed (2010-2015)': {'Blank(s)': 0},
  'Derived AJCC M, 7th ed (2010-2015)': {'Blank(s)': 0},
  'Derived AJCC Stage Group, 6th ed (2004-2015)': {'Blank(s)': 0},
  'Derived AJCC T, 6th ed (2004-2015)': {'Blank(s)': 0},
  'Derived AJCC N, 6th ed (2004-2015)': {'Blank(s)': 0},
  'Derived AJCC M, 6th ed (2004-2015)': {'Blank(s)': 0},
  'AJCC stage 3rd edition (1988-2003)': {'Blank(s)': 0},
  'T value - based on AJCC 3rd (1988-2003)': {'Blank(s)': 0},
  'N value - based on AJCC 3rd (1988-2003)': {'Blank(s)': 0},
  'M value - based on AJCC 3rd (1988-2003)': {'Blank(s)': 0},
  'RX Summ--Surg Prim Site (1998+)': {'Blank(s)': 0},
  'Radiation sequence with surgery': {'Blank(s)': 0},
  'Reason no cancer-directed surgery': {'Blank(s)': 0},
  'Radiation recode': {'None/Unknown': 0},
  'Chemotherapy recode (yes, no/unk)': {'No/Unknown': 0},
  'Regional nodes examined (1988+)': {'Blank(s)': 0},
  'Regional nodes positive (1988+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-bone (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-brain (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-liver (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-lung (2010+)': {'Blank(s)': 0},
  'CS tumor size (2004-2015)': {'Blank(s)': 0},
  'CS extension (2004-2015)': {'Blank(s)': 0},
  'CS lymph nodes (2004-2015)': {'Blank(s)': 0},
  'CS mets at dx (2004-2015)': {'Blank(s)': 0},
  'CS site-specific factor 2 (2004+ varying by schema)': {'Blank(s)': 0},
  'COD to site recode': {'Blank(s)': 0},
  'SEER cause-specific death classification': {'Blank(s)': 0},
  'SEER other cause of death classification': {'Blank(s)': 0},
  'Survival months': {'Blank(s)': 0},
  'Vital status recode (study cutoff used)': {'Blank(s)': 0},
  'Total number of in situ/malignant tumors for patient': {'Blank(s)': 0},
  'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': {'Blank(s)': 0},
  'Age at diagnosis': {'Blank(s)': 0},
  'Insurance Recode (2007+)': {'Blank(s)': 0},
  'Marital status at diagnosis': {'Blank(s)': 0}
};

def read(filename):

  book = xlrd.open_workbook(filename);
  sh = book.sheet_by_index(0);
  headers = sh.row(0);
  samples = list();
  for rx in range(1, sh.nrows):
    sample = dict();
    sex, yd, pslabel, grade, laterality, ss, dasg7, dat7, dan7, dam7, dasg6, dat6, dan6, dam6, as3e, tvba3, nvba3, mvba3, \
      rssps, rsws, rncds, rr, cr, rne, rnp, scmd1, scmd2, scmd3, scmd4, cts, ce, cln, cmad, cssf, csr, scsdc, socdc, sm, vsr, \
      tnistp, ror, aad, ir, msad, patient_id = sh.row(rx);
    if sex.value not in labels_collection['Sex']: labels_collection['Sex'][sex.value] = len(labels_collection['Sex']);
    if pslabel.value not in labels_collection['Primary Site - labeled']: labels_collection['Primary Site - labeled'][pslabel.value] = len(labels_collection['Primary Site - labeled']);
    if grade.value not in labels_collection['Grade']: labels_collection['Grade'][grade.value] = len(labels_collection['Grade']);
    if laterality.value not in labels_collection['Laterality']: labels_collection['Laterality'][laterality.value] = len(labels_collection['Laterality']);
    if ss.value not in labels_collection['Summary stage 2000 (1998+)']: labels_collection['Summary stage 2000 (1998+)'][ss.value] = len(labels_collection['Summary stage 2000 (1998+)']);
    if dasg7.value not in labels_collection['Derived AJCC Stage Group, 7th ed (2010-2015)']: labels_collection['Derived AJCC Stage Group, 7th ed (2010-2015)'][dasg7.value] = len(labels_collection['Derived AJCC Stage Group, 7th ed (2010-2015)']);
    if dat7.value not in labels_collection['Derived AJCC T, 7th ed (2010-2015)']: labels_collection['Derived AJCC T, 7th ed (2010-2015)'][dat7.value] = len(labels_collection['Derived AJCC T, 7th ed (2010-2015)']);
    if dan7.value not in labels_collection['Derived AJCC N, 7th ed (2010-2015)']: labels_collection['Derived AJCC N, 7th ed (2010-2015)'][dan7.value] = len(labels_collection['Derived AJCC N, 7th ed (2010-2015)']);
    if dam7.value not in labels_collection['Derived AJCC M, 7th ed (2010-2015)']: labels_collection['Derived AJCC M, 7th ed (2010-2015)'][dam7.value] = len(labels_collection['Derived AJCC M, 7th ed (2010-2015)']);
    if dasg6.value not in labels_collection['Derived AJCC Stage Group, 6th ed (2004-2015)']: labels_collection['Derived AJCC Stage Group, 6th ed (2004-2015)'][dasg6.value] = len(labels_collection['Derived AJCC Stage Group, 6th ed (2004-2015)']);
    if dat6.value not in labels_collection['Derived AJCC T, 6th ed (2004-2015)']: labels_collection['Derived AJCC T, 6th ed (2004-2015)'][dat6.value] = len(labels_collection['Derived AJCC T, 6th ed (2004-2015)']);
    if dan6.value not in labels_collection['Derived AJCC N, 6th ed (2004-2015)']: labels_collection['Derived AJCC N, 6th ed (2004-2015)'][dan6.value] = len(labels_collection['Derived AJCC N, 6th ed (2004-2015)']);
    if dam6.value not in labels_collection['Derived AJCC M, 6th ed (2004-2015)']: labels_collection['Derived AJCC M, 6th ed (2004-2015)'][dam6.value] = len(labels_collection['Derived AJCC M, 6th ed (2004-2015)']);
    if as3e.value not in labels_collection['AJCC stage 3rd edition (1988-2003)']: labels_collection['AJCC stage 3rd edition (1988-2003)'][as3e.value] = len(labels_collection['AJCC stage 3rd edition (1988-2003)']);
    if tvba3.value not in labels_collection['T value - based on AJCC 3rd (1988-2003)']: labels_collection['T value - based on AJCC 3rd (1988-2003)'][tvba3.value] = len(labels_collection['T value - based on AJCC 3rd (1988-2003)']);
    if nvba3.value not in labels_collection['N value - based on AJCC 3rd (1988-2003)']: labels_collection['N value - based on AJCC 3rd (1988-2003)'][nvba3.value] = len(labels_collection['N value - based on AJCC 3rd (1988-2003)']);
    if mvba3.value not in labels_collection['M value - based on AJCC 3rd (1988-2003)']: labels_collection['M value - based on AJCC 3rd (1988-2003)'][mvba3.value] = len(labels_collection['M value - based on AJCC 3rd (1988-2003)']);
    if rssps.value not in labels_collection['RX Summ--Surg Prim Site (1998+)']: labels_collection['RX Summ--Surg Prim Site (1998+)'][rssps.value] = len(labels_collection['RX Summ--Surg Prim Site (1998+)']);
    if rsws.value not in labels_collection['Radiation sequence with surgery']: labels_collection['Radiation sequence with surgery'][rsws.value] = len(labels_collection['Radiation sequence with surgery']);
    if rncds.value not in labels_collection['Reason no cancer-directed surgery']: labels_collection['Reason no cancer-directed surgery'][rncds.value] = len(labels_collection['Reason no cancer-directed surgery']);
    if rr.value not in labels_collection['Radiation recode']: labels_collection['Radiation recode'][rr.value] = len(labels_collection['Radiation recode']);
    if cr.value not in labels_collection['Chemotherapy recode (yes, no/unk)']: labels_collection['Chemotherapy recode (yes, no/unk)'][cr.value] = len(labels_collection['Chemotherapy recode (yes, no/unk)']);
    if rne.value not in labels_collection['Regional nodes examined (1988+)']: labels_collection['Regional nodes examined (1988+)'][rne.value] = len(labels_collection['Regional nodes examined (1988+)']);
    if rnp.value not in labels_collection['Regional nodes positive (1988+)']: labels_collection['Regional nodes positive (1988+)'][rnp.value] = len(labels_collection['Regional nodes positive (1988+)']);
    if scmd1.value not in labels_collection['SEER Combined Mets at DX-bone (2010+)']: labels_collection['SEER Combined Mets at DX-bone (2010+)'][scmd1.value] = len(labels_collection['SEER Combined Mets at DX-bone (2010+)']);
    if scmd2.value not in labels_collection['SEER Combined Mets at DX-brain (2010+)']: labels_collection['SEER Combined Mets at DX-brain (2010+)'][scmd2.value] = len(labels_collection['SEER Combined Mets at DX-brain (2010+)']);
    if scmd3.value not in labels_collection['SEER Combined Mets at DX-liver (2010+)']: labels_collection['SEER Combined Mets at DX-liver (2010+)'][scmd3.value] = len(labels_collection['SEER Combined Mets at DX-liver (2010+)']);
    if scmd4.value not in labels_collection['SEER Combined Mets at DX-lung (2010+)']: labels_collection['SEER Combined Mets at DX-lung (2010+)'][scmd4.value] = len(labels_collection['SEER Combined Mets at DX-lung (2010+)']);
    if cts.value not in labels_collection['CS tumor size (2004-2015)']: labels_collection['CS tumor size (2004-2015)'][cts.value] = len(labels_collection['CS tumor size (2004-2015)']);
    if ce.value not in labels_collection['CS extension (2004-2015)']: labels_collection['CS extension (2004-2015)'][ce.value] = len(labels_collection['CS extension (2004-2015)']);
    if cln.value not in labels_collection['CS lymph nodes (2004-2015)']: labels_collection['CS lymph nodes (2004-2015)'][cln.value] = len(labels_collection['CS lymph nodes (2004-2015)']);
    if cmad.value not in labels_collection['CS mets at dx (2004-2015)']: labels_collection['CS mets at dx (2004-2015)'][cmad.value] = len(labels_collection['CS mets at dx (2004-2015)']);
    if cssf.value not in labels_collection['CS site-specific factor 2 (2004+ varying by schema)']: labels_collection['CS site-specific factor 2 (2004+ varying by schema)'][cssf.value] = len(labels_collection['CS site-specific factor 2 (2004+ varying by schema)']);
    if csr.value not in labels_collection['COD to site recode']: labels_collection['COD to site recode'][csr.value] = len(labels_collection['COD to site recode']);
    if scsdc.value not in labels_collection['SEER cause-specific death classification']: labels_collection['SEER cause-specific death classification'][scsdc.value] = len(labels_collection['SEER cause-specific death classification']);
    if socdc.value not in labels_collection['SEER other cause of death classification']: labels_collection['SEER other cause of death classification'][socdc.value] = len(labels_collection['SEER other cause of death classification']);
    if sm.value not in labels_collection['Survival months']: labels_collection['Survival months'][sm.value] = len(labels_collection['Survival months']);
    if vsr.value not in labels_collection['Vital status recode (study cutoff used)']: labels_collection['Vital status recode (study cutoff used)'][vsr.value] = len(labels_collection['Vital status recode (study cutoff used)']);
    if tnistp.value not in labels_collection['Total number of in situ/malignant tumors for patient']: labels_collection['Total number of in situ/malignant tumors for patient'][tnistp.value] = len(labels_collection['Total number of in situ/malignant tumors for patient']);
    if ror.value not in labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)']: labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'][ror.value] = len(labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)']);
    if aad.value not in labels_collection['Age at diagnosis']: labels_collection['Age at diagnosis'][aad.value] = len(labels_collection['Age at diagnosis']);
    if ir.value not in labels_collection['Insurance Recode (2007+)']: labels_collection['Insurance Recode (2007+)'][ir.value] = len(labels_collection['Insurance Recode (2007+)']);
    if msad.value not in labels_collection['Marital status at diagnosis']: labels_collection['Marital status at diagnosis'][msad.value] = len(labels_collection['Marital status at diagnosis']);
    
    sample['Sex'] = labels_collection['Sex'][sex.value];
    sample['Primary Site - labeled'] = labels_collection['Primary Site - labeled'][pslabel.value];
    sample['Grade'] = labels_collection['Grade'][grade.value];
    sample['Laterality'] = labels_collection['Laterality'][laterality.value];
    sample['Summary stage 2000 (1998+)'] = labels_collection['Summary stage 2000 (1998+)'][ss.value];
    sample['Derived AJCC Stage Group, 7th ed (2010-2015)'] = labels_collection['Derived AJCC Stage Group, 7th ed (2010-2015)'][dasg7.value];
    sample['Derived AJCC T, 7th ed (2010-2015)'] = labels_collection['Derived AJCC T, 7th ed (2010-2015)'][dat7.value];
    sample['Derived AJCC N, 7th ed (2010-2015)'] = labels_collection['Derived AJCC N, 7th ed (2010-2015)'][dan7.value];
    sample['Derived AJCC M, 7th ed (2010-2015)'] = labels_collection['Derived AJCC M, 7th ed (2010-2015)'][dam7.value];
    sample['Derived AJCC Stage Group, 6th ed (2004-2015)'] = labels_collection['Derived AJCC Stage Group, 6th ed (2004-2015)'][dasg6.value];
    sample['Derived AJCC T, 6th ed (2004-2015)'] = labels_collection['Derived AJCC T, 6th ed (2004-2015)'][dat6.value];
    sample['Derived AJCC N, 6th ed (2004-2015)'] = labels_collection['Derived AJCC N, 6th ed (2004-2015)'][dan6.value];
    sample['Derived AJCC M, 6th ed (2004-2015)'] = labels_collection['Derived AJCC M, 6th ed (2004-2015)'][dam6.value];
    sample['AJCC stage 3rd edition (1988-2003)'] = labels_collection['AJCC stage 3rd edition (1988-2003)'][as3e.value];
    sample['T value - based on AJCC 3rd (1988-2003)'] = labels_collection['T value - based on AJCC 3rd (1988-2003)'][tvba3.value];
    sample['N value - based on AJCC 3rd (1988-2003)'] = labels_collection['N value - based on AJCC 3rd (1988-2003)'][nvba3.value];
    sample['M value - based on AJCC 3rd (1988-2003)'] = labels_collection['M value - based on AJCC 3rd (1988-2003)'][mvba3.value];
    sample['RX Summ--Surg Prim Site (1998+)'] = labels_collection['RX Summ--Surg Prim Site (1998+)'][rssps.value];
    sample['Radiation sequence with surgery'] = labels_collection['Radiation sequence with surgery'][rsws.value];
    sample['Reason no cancer-directed surgery'] = labels_collection['Reason no cancer-directed surgery'][rncds.value];
    sample['Radiation recode'] = labels_collection['Radiation recode'][rr.value];
    sample['Chemotherapy recode (yes, no/unk)'] = labels_collection['Chemotherapy recode (yes, no/unk)'][cr.value];
    sample['Regional nodes examined (1988+)'] = labels_collection['Regional nodes examined (1988+)'][rne.value];
    sample['Regional nodes positive (1988+)'] = labels_collection['Regional nodes positive (1988+)'][rnp.value];
    sample['SEER Combined Mets at DX-bone (2010+)'] = labels_collection['SEER Combined Mets at DX-bone (2010+)'][scmd1.value];
    sample['SEER Combined Mets at DX-brain (2010+)'] = labels_collection['SEER Combined Mets at DX-brain (2010+)'][scmd2.value];
    sample['SEER Combined Mets at DX-liver (2010+)'] = labels_collection['SEER Combined Mets at DX-liver (2010+)'][scmd3.value];
    sample['SEER Combined Mets at DX-lung (2010+)'] = labels_collection['SEER Combined Mets at DX-lung (2010+)'][scmd4.value];
    sample['CS tumor size (2004-2015)'] = labels_collection['CS tumor size (2004-2015)'][cts.value];
    sample['CS extension (2004-2015)'] = labels_collection['CS extension (2004-2015)'][ce.value];
    sample['CS lymph nodes (2004-2015)'] = labels_collection['CS lymph nodes (2004-2015)'][cln.value];
    sample['CS mets at dx (2004-2015)'] = labels_collection['CS mets at dx (2004-2015)'][cmad.value];
    sample['CS site-specific factor 2 (2004+ varying by schema)'] = labels_collection['CS site-specific factor 2 (2004+ varying by schema)'][cssf.value];
    sample['COD to site recode'] = labels_collection['COD to site recode'][csr.value];
    sample['SEER cause-specific death classification'] = labels_collection['SEER cause-specific death classification'][scsdc.value];
    sample['SEER other cause of death classification'] = labels_collection['SEER other cause of death classification'][socdc.value];
    sample['Survival months'] = labels_collection['Survival months'][sm.value];
    sample['Vital status recode (study cutoff used)'] = labels_collection['Vital status recode (study cutoff used)'][vsr.value];
    sample['Total number of in situ/malignant tumors for patient'] = labels_collection['Total number of in situ/malignant tumors for patient'][tnistp.value];
    sample['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'] = labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'][ror.value];
    sample['Age at diagnosis'] = labels_collection['Age at diagnosis'][aad.value];
    sample['Insurance Recode (2007+)'] = labels_collection['Insurance Recode (2007+)'][ir.value];
    sample['Marital status at diagnosis'] = labels_collection['Marital status at diagnosis'][msad.value];
    samples.append(sample);
  # delete column with only one possible value
  keys = list();
  for key, labels in labels_collection.items():
    if labels is not None and len(labels) == 2:
      keys.append(key);
      for sample in samples:
        del sample[key];
  for key in keys: del labels_collection[key];
  with open('dict.pkl', 'wb') as f:
    f.write(pickle.dumps(labels_collection));
  with open('dataset.pkl', 'wb') as f:
    f.write(pickle.dumps(samples));

if __name__ == "__main__":

  read('lung atypical carcinoid.xlsx');

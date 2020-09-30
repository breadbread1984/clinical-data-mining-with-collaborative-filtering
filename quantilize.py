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
  'RX Summ--Surg Prim Site (1998+)': None, # -1 represents blank
  'Radiation sequence with surgery': {'Blank(s)': 0},
  'Reason no cancer-directed surgery': {'Blank(s)': 0},
  'Radiation recode': {'None/Unknown': 0},
  'Chemotherapy recode (yes, no/unk)': {'No/Unknown': 0},
  'Regional nodes examined (1988+)': None, # None represents blank
  'Regional nodes positive (1988+)': None, # None represents blank
  'SEER Combined Mets at DX-bone (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-brain (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-liver (2010+)': {'Blank(s)': 0},
  'SEER Combined Mets at DX-lung (2010+)': {'Blank(s)': 0},
  'CS tumor size (2004-2015)': None, # None represents blank
  'CS extension (2004-2015)': None, # None represents blank
  'CS lymph nodes (2004-2015)': None, # None represents blank
  'CS mets at dx (2004-2015)': None, # None represents blank
  'CS site-specific factor 2 (2004+ varying by schema)': None, # None represents blank
  'COD to site recode': {'Blank(s)': 0},
  'SEER cause-specific death classification': {'Blank(s)': 0},
  'SEER other cause of death classification': {'Blank(s)': 0},
  'Survival months': None, # None represents blank
  'Vital status recode (study cutoff used)': {'Blank(s)': 0},
  'Total number of in situ/malignant tumors for patient': None, # None represents blank
  'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': {'Blank(s)': 0},
  'Age at diagnosis': None, # None represents blank
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
    if rsws.value not in labels_collection['Radiation sequence with surgery']: labels_collection['Radiation sequence with surgery'][rsws.value] = len(labels_collection['Radiation sequence with surgery']);
    if rncds.value not in labels_collection['Reason no cancer-directed surgery']: labels_collection['Reason no cancer-directed surgery'][rncds.value] = len(labels_collection['Reason no cancer-directed surgery']);
    if rr.value not in labels_collection['Radiation recode']: labels_collection['Radiation recode'][rr.value] = len(labels_collection['Radiation recode']);
    if cr.value not in labels_collection['Chemotherapy recode (yes, no/unk)']: labels_collection['Chemotherapy recode (yes, no/unk)'][cr.value] = len(labels_collection['Chemotherapy recode (yes, no/unk)']);
    if scmd1.value not in labels_collection['SEER Combined Mets at DX-bone (2010+)']: labels_collection['SEER Combined Mets at DX-bone (2010+)'][scmd1.value] = len(labels_collection['SEER Combined Mets at DX-bone (2010+)']);
    if scmd2.value not in labels_collection['SEER Combined Mets at DX-brain (2010+)']: labels_collection['SEER Combined Mets at DX-brain (2010+)'][scmd2.value] = len(labels_collection['SEER Combined Mets at DX-brain (2010+)']);
    if scmd3.value not in labels_collection['SEER Combined Mets at DX-liver (2010+)']: labels_collection['SEER Combined Mets at DX-liver (2010+)'][scmd3.value] = len(labels_collection['SEER Combined Mets at DX-liver (2010+)']);
    if scmd4.value not in labels_collection['SEER Combined Mets at DX-lung (2010+)']: labels_collection['SEER Combined Mets at DX-lung (2010+)'][scmd4.value] = len(labels_collection['SEER Combined Mets at DX-lung (2010+)']);
    if csr.value not in labels_collection['COD to site recode']: labels_collection['COD to site recode'][csr.value] = len(labels_collection['COD to site recode']);
    if scsdc.value not in labels_collection['SEER cause-specific death classification']: labels_collection['SEER cause-specific death classification'][scsdc.value] = len(labels_collection['SEER cause-specific death classification']);
    if socdc.value not in labels_collection['SEER other cause of death classification']: labels_collection['SEER other cause of death classification'][socdc.value] = len(labels_collection['SEER other cause of death classification']);
    if vsr.value not in labels_collection['Vital status recode (study cutoff used)']: labels_collection['Vital status recode (study cutoff used)'][vsr.value] = len(labels_collection['Vital status recode (study cutoff used)']);
    if ror.value not in labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)']: labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'][ror.value] = len(labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)']);
    if ir.value not in labels_collection['Insurance Recode (2007+)']: labels_collection['Insurance Recode (2007+)'][ir.value] = len(labels_collection['Insurance Recode (2007+)']);
    if msad.value not in labels_collection['Marital status at diagnosis']: labels_collection['Marital status at diagnosis'][msad.value] = len(labels_collection['Marital status at diagnosis']);
    
    sample['Sex'] = labels_collection['Sex'][sex.value] if sex.value != 'Blank(s)' else None;
    sample['Primary Site - labeled'] = labels_collection['Primary Site - labeled'][pslabel.value] if pslabel.value != 'Blank(s)' else None;
    sample['Grade'] = labels_collection['Grade'][grade.value] if grade.value != 'Blank(s)' else None;
    sample['Laterality'] = labels_collection['Laterality'][laterality.value] if laterality.value != 'Blank(s)' else None;
    sample['Summary stage 2000 (1998+)'] = labels_collection['Summary stage 2000 (1998+)'][ss.value] if ss.value != 'Blank(s)' else None;
    sample['Derived AJCC Stage Group, 7th ed (2010-2015)'] = labels_collection['Derived AJCC Stage Group, 7th ed (2010-2015)'][dasg7.value] if dasg7.value != 'Blank(s)' else None;
    sample['Derived AJCC T, 7th ed (2010-2015)'] = labels_collection['Derived AJCC T, 7th ed (2010-2015)'][dat7.value] if dat7.value != 'Blank(s)' else None;
    sample['Derived AJCC N, 7th ed (2010-2015)'] = labels_collection['Derived AJCC N, 7th ed (2010-2015)'][dan7.value] if dan7.value != 'Blank(s)' else None;
    sample['Derived AJCC M, 7th ed (2010-2015)'] = labels_collection['Derived AJCC M, 7th ed (2010-2015)'][dam7.value] if dam7.value != 'Blank(s)' else None;
    sample['Derived AJCC Stage Group, 6th ed (2004-2015)'] = labels_collection['Derived AJCC Stage Group, 6th ed (2004-2015)'][dasg6.value] if dasg6.value != 'Blank(s)' else None;
    sample['Derived AJCC T, 6th ed (2004-2015)'] = labels_collection['Derived AJCC T, 6th ed (2004-2015)'][dat6.value] if dat6.value != 'Blank(s)' else None;
    sample['Derived AJCC N, 6th ed (2004-2015)'] = labels_collection['Derived AJCC N, 6th ed (2004-2015)'][dan6.value] if dan6.value != 'Blank(s)' else None;
    sample['Derived AJCC M, 6th ed (2004-2015)'] = labels_collection['Derived AJCC M, 6th ed (2004-2015)'][dam6.value] if dam6.value != 'Blank(s)' else None;
    sample['AJCC stage 3rd edition (1988-2003)'] = labels_collection['AJCC stage 3rd edition (1988-2003)'][as3e.value] if as3e.value != 'Blank(s)' else None;
    sample['T value - based on AJCC 3rd (1988-2003)'] = labels_collection['T value - based on AJCC 3rd (1988-2003)'][tvba3.value] if tvba3.value != 'Blank(s)' else None;
    sample['N value - based on AJCC 3rd (1988-2003)'] = labels_collection['N value - based on AJCC 3rd (1988-2003)'][nvba3.value] if nvba3.value != 'Blank(s)' else None;
    sample['M value - based on AJCC 3rd (1988-2003)'] = labels_collection['M value - based on AJCC 3rd (1988-2003)'][mvba3.value] if mvba3.value != 'Blank(s)' else None;
    sample['RX Summ--Surg Prim Site (1998+)'] = int(rssps.value) if rssps.value != 'Blank(s)' else None;
    sample['Radiation sequence with surgery'] = labels_collection['Radiation sequence with surgery'][rsws.value] if rsws.value != 'Blank(s)' else None;
    sample['Reason no cancer-directed surgery'] = labels_collection['Reason no cancer-directed surgery'][rncds.value] if rncds.value != 'Blank(s)' else None;
    sample['Radiation recode'] = labels_collection['Radiation recode'][rr.value] if rr.value != 'None/Unknown' else None;
    sample['Chemotherapy recode (yes, no/unk)'] = labels_collection['Chemotherapy recode (yes, no/unk)'][cr.value] if cr.value != 'No/Unknown' else None;
    sample['Regional nodes examined (1988+)'] = int(rne.value) if rne.value != 'Blank(s)' else None;
    sample['Regional nodes positive (1988+)'] = int(rnp.value) if rnp.value != 'Blank(s)' else None;
    sample['SEER Combined Mets at DX-bone (2010+)'] = labels_collection['SEER Combined Mets at DX-bone (2010+)'][scmd1.value] if scmd1.value != 'Blank(s)' else None;
    sample['SEER Combined Mets at DX-brain (2010+)'] = labels_collection['SEER Combined Mets at DX-brain (2010+)'][scmd2.value] if scmd2.value != 'Blank(s)' else None;
    sample['SEER Combined Mets at DX-liver (2010+)'] = labels_collection['SEER Combined Mets at DX-liver (2010+)'][scmd3.value] if scmd3.value != 'Blank(s)' else None;
    sample['SEER Combined Mets at DX-lung (2010+)'] = labels_collection['SEER Combined Mets at DX-lung (2010+)'][scmd4.value] if scmd4.value != 'Blank(s)' else None;
    sample['CS tumor size (2004-2015)'] = int(cts.value) if cts.value != 'Blank(s)' else None;
    sample['CS extension (2004-2015)'] = int(ce.value) if ce.value != 'Blank(s)' else None;
    sample['CS lymph nodes (2004-2015)'] = int(cln.value) if cln.value != 'Blank(s)' else None;
    sample['CS mets at dx (2004-2015)'] = int(cmad.value) if cmad.value != 'Blank(s)' else None;
    sample['CS site-specific factor 2 (2004+ varying by schema)'] = int(cssf.value) if cssf.value != 'Blank(s)' else None;
    sample['COD to site recode'] = labels_collection['COD to site recode'][csr.value] if csr.value != 'Blank(s)' else None;
    sample['SEER cause-specific death classification'] = labels_collection['SEER cause-specific death classification'][scsdc.value] if scsdc.value != 'Blank(s)' else None;
    sample['SEER other cause of death classification'] = labels_collection['SEER other cause of death classification'][socdc.value] if socdc.value != 'Blank(s)' else None;
    sample['Survival months'] = int(sm.value) if sm.value != 'Blank(s)' else None;
    sample['Vital status recode (study cutoff used)'] = labels_collection['Vital status recode (study cutoff used)'][vsr.value] if vsr.value != 'Blank(s)' else None;
    sample['Total number of in situ/malignant tumors for patient'] = int(tnistp.value) if tnistp.value != 'Blank(s)' else None;
    sample['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'] = labels_collection['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'][ror.value] if ror.value != 'Blank(s)' else None;
    sample['Age at diagnosis'] = int(aad.value) if aad.value != 'Blank(s)' else None;
    sample['Insurance Recode (2007+)'] = labels_collection['Insurance Recode (2007+)'][ir.value] if ir.value != 'Blank(s)' else None;
    sample['Marital status at diagnosis'] = labels_collection['Marital status at diagnosis'][msad.value] if msad.value != 'Blank(s)' else None;
    samples.append(sample);
  with open('dict.pkl', 'wb') as f:
    f.write(pickle.dumps(labels_collection));
  with open('dataset.pkl', 'wb') as f:
    f.write(pickle.dumps(samples));

if __name__ == "__main__":

  read('lung atypical carcinoid.xlsx');

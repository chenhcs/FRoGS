#!/usr/bin/env python
#QueryPertID,GeneID,grp,Pct,KnownTarget,y,Structure_RefPertID,Structure_Similarity,Best_Structure_RefPertID,Best_Structure_Similarity,PSP_QuerySID,PSP_RefSID,PSP_r,PSP_sim,CLiP_RefPertID,CLiP_r,CLiP_sim,NCI60_RefPertID,NCI60_r,NCI60_sim,pQSARCorr_QueryNVP,pQSARCorr_RefNVP,pQSARCorr_r,pQSAR_sim,Best_pQSARCorr_QueryNVP,Best_pQSARCorr_RefNVP,Best_pQSARCorr_r,Best_pQSAR_sim,pQSARCorr_x_QueryNVP,pQSARCorr_x_RefNVP,pQSARCorr_x_r,pQSAR_sim_x,pQSARpIC50_pIC50,pQSARpIC50_x_pIC50,CDFCorr_QueryNVP,CDFCorr_RefNVP,CDFCorr_r,CDF_sim,CDFCorr_x_QueryNVP,CDFCorr_x_RefNVP,CDFCorr_x_r,CDF_sim_x,CDFpIC50_x_pIC50,CDFpIC50_pIC50,Probability_L,Probability_Q,Probability_Best_Q,Probability_S,Probability_SAR,Probability_PSP,Probability_NCI60,Probability_CLiP,Probability_LQ_nb,Probability_LQ_nbclip,Probability_LQ,Probability_LQS,Probability_LQSAR,Order_L,Order_Q,Order_S,Order_SAR,Order_PSP,Order_NCI60,Order_CLiP,Order_LQ,Order_LQS,Order_LQSAR,Order_LQ_nb,Order_LQ_nbclip,Order_Best_Q,QuerySID,QueryNVP,cpd_description,cpd_name,cpd_broad,Symbol,gene_description,Validation,Validation_Code,Validation_L1000,Validation_pQSAR,Validation_NCI60,Validation_PSP,Validation_StructureLikely,Validation_StructureSure,Validation_pIC50,Validation_Known,Group_Model,Group_Source
import pandas as pd
import numpy as np
import util

class LR:
    def __init__(self, name):
        assert(name in ("L","Q","LQ","NCI60","PSP"))
        import pickle
        with open(f'model/{name}.model.pickle', 'rb') as f:
            self.wt=pickle.load(f)[0]
        self.name=name

    def run(self, X):
        mask=np.isnan(X)
        y_pred=X*self.wt[1]+self.wt[0]
        y_pred[mask]=0
        prob=1.0/(1.0+np.exp(-y_pred))
        return prob

    def rank2logit(self, X):
        mask=np.isnan(X)
        X=np.nan_to_num(X)
        X=np.clip(X, 1e-4, 1-1e-4)
        X=np.log10((1-X)/X)
        X[mask]=np.nan
        return X

    def prob2logit(self, X):
        return -np.log(1/X-1)

class L(LR):
    def __init__(self):
        super().__init__("L")

    def run(self, X):
        return super().run(self.rank2logit(X))

class Q(LR):
    def __init__(self):
        super().__init__("Q")

class LQ(LR):
    def __init__(self):
        super().__init__("LQ")

    def run(self, X):
        if X.ndim==1: X=X.reshape(-1, 2)
        prob_L=L().run(X[:,0])
        prob_Q=Q().run(X[:,1])
        R1=self.prob2logit(prob_L)
        R2=self.prob2logit(prob_Q)
        y_pred=R1*self.wt[1]+R2*self.wt[2]+self.wt[0]
        prob=1.0/(1.0+np.exp(-y_pred))
        return prob

class NCI60(LR):
    def __init__(self):
        super().__init__("NCI60")

class PSP(LR):
    def __init__(self):
        super().__init__("PSP")

if __name__=="__main__":
    t=pd.read_csv('dataset_model.csv')

    t['_Probability_L']=L().run(t[['Pct']].values)
    t['_Probability_Q']=Q().run(t[['pQSARCorr_r']].values)
    t['_Probability_LQ']=LQ().run(t[['Pct','pQSARCorr_r']].values)
    t['_Probability_NCI60']=NCI60().run(t[['NCI60_r']].values)
    t['_Probability_PSP']=PSP().run(t[['PSP_r']].values)

    assert(np.allclose(t['Probability_L'].values, t['_Probability_L'].values, rtol=1e-3))
    assert(np.allclose(t['Probability_Q'].values, t['_Probability_Q'].values, rtol=1e-3))
    assert(np.allclose(t['Probability_LQ'].values, t['_Probability_LQ'].values, rtol=1e-3))
    assert(np.allclose(t['Probability_NCI60'].values, t['_Probability_NCI60'].values, rtol=1e-3))
    assert(np.allclose(t['Probability_PSP'].values, t['_Probability_PSP'].values, rtol=1e-3))

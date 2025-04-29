
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS","UCR", "AIOPS", "NAB", "Yahoo", "WSD","NEK", "GutenTAG", "NormA","CalIt2"]
    # datasets = ["TODS"]
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="../../datasets-main",
        datasets=datasets,
    )
    
   
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    
    from EasyTSAD.Methods import SubLOF, SubOCSVM, AR, LSTMADalpha, LSTMADbeta, AE, EncDecAD, SRCNN, AnomalyTransformer, TFAD, Donut, FCVAE, TimesNet, OFA, FITS

    # Old libraries with dependency issues are excluded: 'MatrixProfile',"SAND",'DCdetector','TFAD','OFA'
    methods = ['SubLOF','SubOCSVM','AR', 'LSTMADalpha', 'LSTMADbeta', 'AE', 'EncDecAD', 'SRCNN','AnomalyTransformer','Donut','FCVAE','TimesNet', 'FITS']

    training_schema = "naive"
    
#     for method in methods:
#         # run models
#         gctrl.run_exps(
#             method=method,
#             training_schema=training_schema
#         )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA()
        ]
    )

    for method in methods:
        gctrl.do_evals(
            method=method,
            training_schema=training_schema
        )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    for method in methods:
        gctrl.plots(
            method=method,
            training_schema=training_schema
        )

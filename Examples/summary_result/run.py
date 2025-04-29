from EasyTSAD.Controller import TSADController

def run_only_once(gctrl, methods, training_schema):
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    
    for method in methods:
        # run models
        gctrl.run_exps(
            method=method,
            training_schema=training_schema
        )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
        ]
    )

    for method in methods:
        gctrl.do_evals(
            method=method,
            training_schema=training_schema
        )
        
        
if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS","UCR", "AIOPS", "NAB", "Yahoo", "WSD","NEK", "GutenTAG", "NormA","CalIt2"]
    dataset_types = "UTS"
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type=dataset_types,
        dirname="../../datasets-main",
        datasets=datasets,
    )
    
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    
    from EasyTSAD.Methods import SubLOF, SubOCSVM, AR, LSTMADalpha, LSTMADbeta, AE, EncDecAD, SRCNN, AnomalyTransformer, TFAD, Donut, FCVAE, TimesNet, OFA, FITS

    # Old libraries with dependency issues are excluded: 'MatrixProfile',"SAND",'DCdetector','TFAD','OFA'
    methods = ['SubLOF','SubOCSVM','AR', 'LSTMADalpha', 'LSTMADbeta', 'AE', 'EncDecAD', 'SRCNN','AnomalyTransformer','Donut','FCVAE','TimesNet', 'FITS']

    training_schema = "naive"
    
    # If your have run this function before and haven't changed the params below,
    # you can skip this step just to get evaluations.

    # run_only_once(gctrl=gctrl, methods=methods, training_schema=training_schema)


    """============= [Aggregation Plots] ============="""
    gctrl.summary.plot_aggreY(
        types=dataset_types,
        datasets=datasets,
        methods=methods,
        training_schema=training_schema
    )
    
    """============= Generate CSVs ============="""
    gctrl.summary.to_csv(
            datasets=datasets,
            methods=methods,
            training_schema=training_schema,
            eval_items=[
                ["best f1 under pa", "f1"],   
            ]
        )
    

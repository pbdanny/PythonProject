# FactorAnalysis

__Correlation matrix.__  
- observed correlation matrix.  
- reproduced correlation matrix.  
- residual correlation matrix.  

__Rotation factor.__
1)  Orthogonal.
    _loading matrix_ : correlation between observed variable vs factor.
2)  Oblique. 
    _factor correlation matrix_ : correlation between factor vs factor.  
    _structure matrix_ : correlation variable vs factor.  
    _pattern matrix_ : correlation each factors and each variable.  

__factor-score coefficient matrix__. 

PCA product component. : observed variable ->(caused) components.  
FA produce factor. : factor ->(caused) observed variable.  

PCA all varience in observed variable analized.  
FA shared varience is analized.  

__Step to compute FA/PCA__.  

1)  Find eigenvalue, eigenvector (Diagonalization) of Covarience matrix (or Correlation matrix). Eigenvalues = re-package of correlation (covarience) matrix.
2)  Correlation matrix = Eigen Vactor x Eigen Value x Eigen Vactor'. 
    Correlation matrix = Eigen Vactor x sqrt(Eigen Value) x sqrt(Eigen Value) x Eigen Vactor'.  
    Eigen Vector x sqrt(Eigen Value) -> Factor Loading.  
    Correlation matrix = Factor loading x Factor loading'.  
    
_Factor Loading_ = correlation between factor and variable, First column = First factor , each rows = correlation with each variables

__Orthogonal Rotation__
maximized correlation factor <-> variable  

_varimax_ high factor loading -> higher / low factor loadin -> lower
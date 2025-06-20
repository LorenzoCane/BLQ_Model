import numpy as np

#***************************************************************************
def rmsd(model, x_data, y_data, **params):
    '''
      Compute the Root Mean Squared Error (RMSD) between a given model and observed data.
    
      Parameters:
      ----------

      model: callable
         The model function. It should accept the independent variable (x_data) as its first argument
            and then any additional parameters as keyword arguments.
      x_data: array-like
         The independent variable data (input data).
      y_data: array-like
         The observed data to compare against.
      **params: dict
         Additional keyword arguments passed to the `model` function (e.g., model parameters).
    
      Returns:
      -------
      float : The root mean squared deviation (RMSD) between model predictions and actual `y_data`.
    '''
    
    # Use list comprehension to compute the model predictions for each x-data point.
    predictions = np.array([model(x, **params) for x in x_data])
    # Compute RMSD.
    rmsd_value = ((np.mean((predictions - y_data) ** 2))**0.5)
    
    return rmsd_value
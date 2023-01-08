def forecast_plot(dates, forecast, target, model):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,6))
    plt.plot(dates, forecast, label = "Forecast")
    # plotting the line 2 points 
    plt.plot(dates, target, label = "Actual")
    plt.xlabel('Date')
    # Set the y axis label of the current axis.
    plt.ylabel('BEIR')
    # Set a title of the current axes.
    plt.title('XGBoost Forecast Inflation vs Actual Inflation')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylim(2.5,3)
    # show a legend on the plot
    plt.legend()
    plt.savefig(model+'_forecast.png')
    # Display a figure.
    plt.show()
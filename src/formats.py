def highlight_values(df):
    """highlight values over 5 and 15 percent of daily value"""

    conditional_format_low = [
    {
        'if': {
            'filter_query': '{{{col}}} > 0.05 && {{{col}}} < 0.15'.format(col=col),
            'column_id': col
        },
        'backgroundColor': 'skyblue',
        'color':'white',
        'fontWeight': 'bold'
    }
    for col in df.columns if col not in ['Nutrient','Daily value']]


    conditional_format_high = [
    {
        'if': {
            'filter_query': '{{{col}}} > 0.15'.format(col=col),
            'column_id': col
        },
        'backgroundColor': 'steelblue',
        'color':'white',
        'fontWeight': 'bold'
    }
    for col in df.columns if col not in ['Nutrient','Daily value']]
    
    return conditional_format_low + conditional_format_high
# End-of-file (EOF)
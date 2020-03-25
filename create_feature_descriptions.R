create_features_description = function(data_table, description_column_number){

    # @param:
    # data_table(tbl, data.frame) table that we will use for machine learning model later
    # description_column_number(numeric): column that contains the description of products
    
    # returns: tibble with features extracted from description column
    
    # rename description_column_number to "description"    
    colnames(data_table)[description_column_number] = "description"

    unique_elements_desc = data_table %>% 
        # extract only one column as a vector
        pull(description) %>% 
        # collapse in one vector by using space
        str_c(collapse = " ") %>% 
        # separate each in separate vector
        str_split(" ") %>% 
        # unlist splitting values
        unlist() %>% 
        # lower everything
        str_to_lower() %>%
        # get unique values
        unique()
    
    featured_data_table = data_table %>% 
        # bind dataframe with character vector
        cbind(as.list(unique_elements_desc)) %>% 
        # replace quotes with nothing
        rename_all(~ str_replace_all(., '"', "")) %>% 
        # lower description column
        mutate(description = str_to_lower(description))
    
    for(col in unique_elements_desc){
        
        # see if a feature is present in the description
        feature_presence = featured_data_table %>% 
            # select current column
            select(description) %>% 
            # turn into a character vector
            pull() %>% 
            # see if engineered feature is present or not
            str_detect(pattern = col) %>% 
            # turn boolean values into numbers
            as.numeric() 
        
        # assign to each feature a value of 1 if it is present in the 
        # description column, otherwise 0
        featured_data_table[,col] = feature_presence
        
    }
    
    featured_data_table = as_tibble(featured_data_table)
    
    return(featured_data_table)

}
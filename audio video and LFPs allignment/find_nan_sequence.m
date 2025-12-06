%% loops through nan array and finds the number of sequences
function [beginning_nan_idx,ending_nan_idx] = find_nan_sequence(LFP_Nans)
    nan_counter = 1;
    total_chunks = 1;
    beginning_nan_idx(1,total_chunks) = LFP_Nans(1);
    ending_nan_idx = [];

    for nan = 2: length(LFP_Nans)
        % last index
        if(nan == length(LFP_Nans))
            ending_nan_idx(1,total_chunks) = LFP_Nans(nan);
        end

        % check if the next index is a nan
        if (LFP_Nans(nan)-LFP_Nans(nan-1)) == 1
            nan_counter = nan_counter + 1; 
        else
            ending_nan_idx(1,total_chunks) = LFP_Nans(nan-1);
            total_chunks = total_chunks + 1;
            beginning_nan_idx(1,total_chunks) = LFP_Nans(nan);
        end
    end
end
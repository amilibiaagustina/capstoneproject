function [LFP_demeaned_split, LFP_Nans, total_chunks,beginning_nans, end_nans] = LFP_demeaned_split(LFP_demeaned)
%% split data
LFP_Nans = find(isnan(LFP_demeaned));

if(~isempty(LFP_Nans))
    %% loop through Nan data
    [beginning_nans, end_nans] = find_nan_sequence(LFP_Nans);
    
    %% find if any of the nan samples are greater than 20 and chunk the data, or else interpolate
    nan_sequence_difference = end_nans - beginning_nans;
    max_samples_interpolate = 20;
    total_chunks = 1;
    LFP_demeaned_split = {};
    
    for sequence = 1:length(nan_sequence_difference)
        if(nan_sequence_difference(sequence)>max_samples_interpolate)
            % make a new chunk of data
            if(sequence == 1)
                LFP_demeaned_split{1,total_chunks} = LFP_demeaned(1:(beginning_nans(sequence))-1);
                total_chunks = total_chunks + 1;
            else
                LFP_demeaned_split{1,total_chunks} = LFP_demeaned((end_nans(sequence-1)+1):(beginning_nans(sequence))-1);
                total_chunks = total_chunks + 1;
            end
        else
            %interpolate betweeen nan points
            % 20 samples each direction
            interp_window =  LFP_demeaned(((beginning_nans(sequence))-20):(end_nans(sequence)+20));
            non_nan_idx = ~isnan(interp_window);
            Y = cumsum(non_nan_idx-diff([1,X])/2);
            Z = interp1(1:nnz(non_nan_idx),interp_window(non_nan_idx),Y)
            LFP_demeaned(((beginning_nans(sequence))-20):(end_nans(sequence)+20)) = Z;
        end
    
    end
    
    if end_nans(sequence) < length(LFP_demeaned)
        LFP_demeaned_split{1,total_chunks} = LFP_demeaned((end_nans(sequence)+1):length(LFP_demeaned));
    end
else
    total_chunks = 1;
    LFP_demeaned_split{1,1} = LFP_demeaned;
    beginning_nans = 0;
    end_nans = 0;
end
end
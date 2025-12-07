%to obtain the actual timestamps

function curr_timestamps = getCurrentTimestamps(num_chunk, LFP_Nans_Left, timestamps, beginning_nans, end_nans)
    if ~isempty(LFP_Nans_Left)
        if num_chunk == 1
            curr_timestamps = timestamps(1:(beginning_nans(num_chunk))-1);
        elseif (num_chunk == total_chunks_Left)
            curr_timestamps = timestamps((end_nans(num_chunk-1)+1):length(LFP_demeaned_Left));
        else
            curr_timestamps = timestamps((end_nans(num_chunk-1)+1):(beginning_nans(num_chunk))-1);
        end
    else
        curr_timestamps = timestamps;
    end
end
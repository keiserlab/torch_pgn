from pgn.data.dmpnn_utils import BatchProxGraph

def _format_batch(train_args, data):
    if train_args.encoder_type == 'd-mpnn':
        return BatchProxGraph(data)
    else:
        return data
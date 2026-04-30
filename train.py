from Transformer_Model import Model

if __name__ == '__main__':
    data_path = ['data/Dakota', 'data/Dodge', 'data/Hennepin', 'data/Steele']
    save_path = 'models/Transformer/Terra_v1'

    Model.train(data_path, save_path, max_epochs=300, lr=1e-4)
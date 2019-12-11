import urllib.request
import numpy as np
import torch
import torch.backends.cudnn
import json
from DeepFM import DeepFM

class MyModel(object):

    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, fix = 2, url = 'https://shield.mlamp.cn/task/api/file/space/download/b3b84baf1fd59ca01bb3d2e95cde70fe/60288/model.m'):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        self.fix = fix
        self.url = url
        self.loaded = False
        self.model = None

    def load(self):
        print("start download")
        print(self.url)
        urllib.request.urlretrieve(self.url, "model.m")
        print("start loading model")
        self.model = torch.load('model.m', map_location=torch.device('cpu'))
        print("model loaded")
        self.loaded = True

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        # parse parameters
        request_size = X[0]
        base_size = X[1]
        base_values = X[2]
        feature_size = X[3]
        feature_values = X[4]
        flag = False
        msg = None

        base_values = np.array(base_values)
        feature_values = np.array(feature_values)

        if base_values.size != base_size:
            flag = True
            msg = 'baseSize check failure'

        if feature_values.size != feature_size:
            flag = True
            msg = 'featureSize check failure'

        if flag:
            return "parameter error %s" % msg

        if not self.loaded:
            self.load()

        if self.model:
            new_array = self.generate_array(base_size, base_values, feature_size, feature_values)
            x1, x2 = self.generate_torch_data(feature_size, new_array)
            t0 = torch.tensor(x1)
            t1 = torch.tensor(x2)
            result = torch.sigmoid(self.model(t0, t1)).data
            result = self.deal_result(result.numpy(), request_size, feature_size, feature_values)
            return result
        else:
            return "less is more more more more %d" % self.fix

    def generate_array(self, base_size, base_values, feature_size, feature_values):
        new_arry = np.zeros((feature_size, base_size))
        for i in range(feature_size):
            new_arry[i] = base_values

        return np.insert(new_arry, 1, feature_values, axis=1)

    def generate_torch_data(self, feature_size, new_array):
        xi = []
        xv = []

        for size in range(feature_size):
            t1, t2 = self.trans(new_array[size])
            xi.append(t1)
            xv.append(t2)

        xi = np.array(xi)
        xv = np.array(xv)
        return xi, xv

    def trans(self, aim58):
        t1 = aim58[0:8]
        xi = np.array([ [t1[0]], [t1[1]], [t1[2]], [t1[3]], [t1[4]], [t1[5]], [t1[6]], [t1[7]] ], dtype='long')
        t2_head = [1, 1, 1, 1, 1, 1, 1, 1 ]
        xv = np.array(t2_head + list(aim58[8:]), dtype='float32')
        return (xi, xv)

    def deal_result(self, result, request_size, feature_size, feature_values):
        result_dict = {}
        for i in range(feature_size):
            result_dict[feature_values[i]] = result[i]

        #result_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        size = len(result_dict)
        if request_size < size:
            return result_dict[:request_size]

        #return json.dumps(self.aggregation_json(result_dict), cls=NpEncoder)
        #json.dumps(self.aggregation_json(result_dict))
        return self.aggregation_json(result_dict)

    def aggregation_json(self, result_dict):
        item_list = []
        for (k, v) in result_dict.items():
            item_dict = {}
            item_dict["id"] = int(k)
            item_dict["score"] = float(v)
            item_list.append(item_dict)

        return item_list

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#tt1 =  [[106028, 195092, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ]]
#tt= [2, 57, [106028, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ], 1, [195092]]
# tt= [2, 57, [106028, 10568, 23, 147, 4, 11, 1241, 6.09887064e-01, -3.17003161e-01, -1.83307350e-01, -4.45917211e-02, -4.00365591e-02,  2.60544335e-03, -2.43274420e-02, -1.35902567e-02, -2.06686687e-02, -2.39776302e-04, -8.98106117e-03, -1.32717369e-02, -9.00286250e-03, -9.20017343e-03, -1.12582045e-02, -9.56592243e-03, -5.72999334e-03, -3.99997272e-03, -9.94744524e-03, -6.57328777e-03, -4.06617252e-03, -7.16522615e-03, -3.39697767e-03, -5.05888509e-03, -6.38805423e-03, -6.68853614e-03, -6.55218540e-03, -3.32565443e-03, -7.25812372e-03, -9.18245874e-04,  1.18093006e-03, 3.55028023e-04, -4.88233333e-03, -1.80893322e-03, -3.13342735e-03, -3.14912642e-03, -4.47223382e-03, 8.49320320e-04, -3.30703938e-03, -3.95207189e-06, -3.04178707e-03, -3.35240504e-03, -2.29544588e-03, -2.08881940e-03, -1.75165117e-03, -2.58994359e-03, 5.19961119e-04, -3.13837733e-03, -3.30228242e-03, 3.50067829e-04 ], 2, [195092, 195093]]
# aa = MyModel()
# tt = np.array(tt)
# print(aa.predict(tt))
#aa.predict(tt)
# for i in range(200):
#     aa.predict(tt)


from boto.s3.connection import Location
import lasagne
import cPickle as pickle

def save(l_out,filename):
    all_params = lasagne.layers.get_all_params(l_out)
    all_params_values = [p.get_value() for p in all_params]

    with open('conv_whale.pickled','w') as f:
        pickle.dump(all_params_values,f)
    save2s3(filename)

def save2s3(filename, bucket_name=None):
    """ Saves the parameters of a model to an S3 bucket."""

    AWS_ACCES_KEY_ID = ''
    AWS_SECRET_ACCES_KEY = ''

    import boto
    #bucket_name  = 'right-whale'
    conn = boto.connect_s3(AWS_ACCES_KEY_ID, 
          AWS_SECRET_ACCES_KEY)

    import boto.s3
    from boto.s3.bucket import Bucket
    if bucket_name == None:
        # bucket = conn.create_bucket(bucket_name,
        #     location = Location.EU)
        bucket = conn.create_bucket(bucket_name)
    else:
        bucket = Bucket(conn, bucket_name)
    testfile = filename
    print 'Uploading %s to Amazon S3 bucket %s' % \
        (testfile,bucket_name)

    import sys
    def percent_cb(complete,total):
        sys.stdout.write('.')
        sys.stdout.flush()

    from boto.s3.key import Key
    k = Key(bucket)
    k.key = filename
    k.set_contents_from_filename(testfile)
    #k.set_contents_from_filename(testfile,
    # cb = percent_cb, num_cb = 10)

def load(l_out,filename): 
    """ Loads parameter values into a model.
    
    Note: You need to build the (compatible) model beforehand. This will only 
    set the values of the weights in the model.
    """ 
    
    with open(filename) as f:
        all_param_values = pickle.load(f)
    
    all_params = lasagne.layers.get_all_params(l_out)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)    

#!/usr/bin/env python3


from argparse import ArgumentParser
import os
import time


from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask_sql import Context, run_server


datadir = 'data/new-york-city/2022-06-03'

SCHEMAS = ['airbnb']

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(
            os.path.dirname(
                __file__
            )
        )
    ),
    'data'
)



def parse_args(argv=None):

    parser = ArgumentParser()

    parser.add_argument(
        '--data-dir', default=DEFAULT_DATA_DIR, metavar='DIR', 
        help='store data under DIR')

    parser.add_argument(
        '--port', default=8080, metavar='PORT',
        help='listen on PORT')

    args = parser.parse_args(argv)

    return args, parser


def uncategorize(data, exceptions=[]):
    """
    Convert categorical dtypes to their underlying dtype.
    """
    data = data.copy()
    for col in data.columns:
        if hasattr(data[col], 'cat') and col not in exceptions:
            dtype = data[col].dtype.categories.dtype
            data[col] = data[col].astype(dtype)
    return data


def main():

    args, parser = parse_args()

    print('Starting cluster...')
    cluster = LocalCluster()
    client = Client(cluster)
    
    print('Registering tables...')
    c = Context()

    # in principle we could organize the data using schemas
    # but it's just easier having all the tables under root
    for schema_name in SCHEMAS:
        #c.create_schema(schema_name)
        schema_dir = os.path.join(args.data_dir, schema_name)
        table_names = os.listdir(schema_dir)
        for table_name in table_names:
            print(f'Registering table "{table_name}" ...')
            data = (
                dd.read_parquet(f'{schema_dir}/{table_name}/', engine='pyarrow')
                .pipe(uncategorize)
            )
            #c.create_table(table_name, data, schema_name=schema_name)
            c.create_table(table_name, data)

    print('Starting server...')
    c.run_server(client)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import subprocess

import numpy as np
import pandas as pd
import dask.dataframe as dd

from dask.distributed import Client, LocalCluster


SOURCE_DOMAIN = f'http://data.insideairbnb.com'

DATASET = 'airbnb'

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
        'specs', nargs='+', metavar='C/S/C/D',
        help='Country/State/City/Date to process')

    args = parser.parse_args(argv)

    args.raw_dir = os.path.join(args.data_dir, 'raw')
    args.dataset_dir = os.path.join(args.data_dir, DATASET)

    return args, parser


def ensure_mirror(filename, *, spec, raw_dir):
    local_filename = os.path.join(raw_dir, filename)
    if os.path.exists(local_filename):
        return local_filename

    remote_filename = f'{SOURCE_DOMAIN}/{spec}/data/{filename}'
    cmd = ['curl', '-O', remote_filename]
    print(' '.join(cmd))

    subprocess.run(cmd)
    os.makedirs(raw_dir, exist_ok=True)
    os.rename(filename, local_filename)
    return local_filename


def read_calendar(spec, raw_dir):
    filename = ensure_mirror('calendar.csv.gz', spec=spec, raw_dir=raw_dir)
    print(f'Loading {filename} ...')
    return (
        pd.read_csv(filename, low_memory=False)
        .assign(
            date = lambda x: pd.to_datetime(x.date, utc=True),
            price = lambda x: x.price.str.replace('[$,]', '', regex=True).astype('float32'),
            adjusted_price = lambda x: x.adjusted_price.str.replace('[$,]', '', regex=True).astype('float32'),
            available = lambda x: x.available.eq('t'),
            minimum_nights = lambda x: x.minimum_nights.astype('float32'),
            maximum_nights = lambda x: x.maximum_nights.astype('float32'),
        )
    )


def read_listings(spec, raw_dir):
    filename = ensure_mirror('listings.csv.gz', spec=spec, raw_dir=raw_dir)
    print(f'Loading {filename} ...')

    raw_listings = (
        pd.read_csv(filename, low_memory=False)
        .assign(
            last_scraped = lambda x: pd.to_datetime(x.last_scraped, utc=True),
            host_since = lambda x: pd.to_datetime(x.host_since, utc=True),
            has_availability = lambda x: x.has_availability.eq('t'),
            calendar_last_scraped = lambda x: pd.to_datetime(x.calendar_last_scraped, utc=True),
            first_review = lambda x: pd.to_datetime(x.first_review, utc=True),
            last_review = lambda x: pd.to_datetime(x.last_review, utc=True),
            instant_bookable = lambda x: x.instant_bookable.eq('t'),
            price = lambda x: x.price.str.replace('[$,]', '', regex=True).astype('float32'),
            host_has_profile_pic = lambda x: x.host_has_profile_pic.eq('t'),
            host_identity_verified = lambda x: x.host_identity_verified.eq('t'),
            host_response_rate = lambda x: (
                x.host_response_rate.fillna('0%')
                .str.replace('%', '', regex=False)
                .astype('float32') / 100),
            host_acceptance_rate = lambda x: (
                x.host_acceptance_rate.fillna('0%')
                .str.replace('%', '', regex=False)
                .astype('float32') / 100),
            host_is_superhost = lambda x: x.host_is_superhost.eq('t'),
            host_has_email_verification = lambda x: x.host_verifications.str.contains("'email'"),
            host_has_phone_verification = lambda x: x.host_verifications.str.contains("'phone'"),
            host_has_work_email_verification = lambda x: x.host_verifications.str.contains("'work_email'"),
        )
        .drop(columns='host_verifications')
    )

    # extract hosts table
    host_orig_columns = pd.Index([
        c for c in raw_listings.columns
        if 'host' in c
    ])
    host_columns = host_orig_columns.str.replace('host_', '', regex=True)
    hosts = (
        raw_listings
        [host_orig_columns]
        .sort_values(['host_id', 'host_since'])
        .groupby('host_id').first().reset_index()
        .rename(columns=dict(zip(host_orig_columns, host_columns)))
        .rename(columns={'id':'host_id'})
        .reset_index(drop=True)
    )

    # extract listings table
    listing_columns = pd.Index([
        c for c in raw_listings.columns
        if c == 'host_id' or c not in host_orig_columns
    ])
    listings = (
        raw_listings[listing_columns]
        .rename(columns={'id':'listing_id'})
        .reset_index(drop=True)
    )

    # clean text fields
    str_columns = listings.dtypes[listings.dtypes.astype('str').eq('object')].index
    for column in str_columns:
        listings[column] = listings[column].fillna('')
        
    # set review fields to float
    review_columns = [c for c in listings.columns if c.startswith('review')]
    for column in review_columns:
        listings[column] = listings[column].astype('float32')

    return listings, hosts


def read_reviews(spec, raw_dir):
    filename = ensure_mirror('reviews.csv.gz', spec=spec, raw_dir=raw_dir)
    print(f'Loading {filename} ...')
    out = (
        pd.read_csv(filename, low_memory=False)
        .assign(
            date = lambda x: pd.to_datetime(x.date),
        )
        .rename(columns={
            'id':'review_id',
            'date':'review_date',
        })
    )
    return out


def write_one(pandas_df, *, npartitions, filename):
    df = dd.from_pandas(pandas_df, npartitions=npartitions)
    print(f'Writing {filename} ...')
    return df.to_parquet(filename, engine='pyarrow')


def process_spec(spec, *, raw_dir, dataset_dir):
    print(f'Processing {spec} ...')
    country, state, city, date_str = spec.split('/')
    spec_raw_dir = os.path.join(raw_dir, spec)

    part_str = os.path.join(
        f'access_date={date_str}',
        f'country={country}',
        f'state={state}',
        f'city={city}',
    )
    parquet_name = lambda name: os.path.join(dataset_dir, name, part_str)

    listings, hosts = read_listings(spec, spec_raw_dir)
    listings.pipe(write_one, npartitions=4, filename=parquet_name('listings'))
    hosts.pipe(write_one, npartitions=1, filename=parquet_name('hosts'))
    del listings, hosts

    reviews = read_reviews(spec, spec_raw_dir)
    reviews.pipe(write_one, npartitions=4, filename=parquet_name('reviews'))
    del reviews

    calendar = read_calendar(spec, spec_raw_dir)
    calendar.pipe(write_one, npartitions=16, filename=parquet_name('calendar'))
    del calendar


def main():
    t0 = pd.to_datetime('today')
    print(f'Start: {t0}')
    args, parser = parse_args()

    print(f'Dataset dir: {args.dataset_dir}')

    for spec in args.specs:
        process_spec(
            spec,
            raw_dir=args.raw_dir,
            dataset_dir=args.dataset_dir,
        )

    t1 = pd.to_datetime('today')
    print(f'End: {t0} ({t1 - t0} elapsed)')


if __name__ == '__main__':
    main()

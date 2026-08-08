def raw_portfolio_eda(df):
    print("portfolio overview")
    print(f"Total raw records: {len(df):,}")
    print("Columns:", df.columns.tolist())
    
    print("missingness summary")
    missing = df.isna().mean().sort_values(ascending=False)
    print(missing[missing > 0])

    print("age distribution")
    if 'age' in df.columns:
        print(df['age'].describe())
        print("Age missing:", df['age'].isna().mean())

    print("gender distribution")
    if 'gender' in df.columns:
        print(df['gender'].value_counts(dropna=False))
        print("Gender missing:", df['gender'].isna().mean())

    print("socioeconomic variables")
    socio_cols = [c for c in df.columns if c.startswith('C_')]
    for col in socio_cols:
        print(f"\n{col}: missing={df[col].isna().mean():.3f}")
        print(df[col].describe())

    print("premium and claims")
    if 'premium' in df.columns:
        print("Premium summary:")
        print(df['premium'].describe())
    if 'cost_claims_year' in df.columns:
        print("\nClaims cost summary:")
        print(df['cost_claims_year'].describe())

    print("policy structure")
    structure_cols = ['policy_type', 'distribution_channel', 'seniority_insured', 'seniority_policy']
    for col in structure_cols:
        if col in df.columns:
            print(f"\n{col}: missing={df[col].isna().mean():.3f}")
            print(df[col].describe())

    print("target variables")
    if 'duration_years' in df.columns:
        print(df['duration_years'].describe())
    if 'event' in df.columns:
        print(df['event'].value_counts(dropna=False))

    print("outliers check")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numeric_cols].describe(percentiles=[0.01, 0.5, 0.99]))

    print("correlation with duration")
    if 'duration_years' in df.columns:
        corr = df[numeric_cols].corr()['duration_years'].sort_values(ascending=False)
        print(corr.head(10))
        print(corr.tail(10))
    
    print("raw eda complete")
    

def exploratory_analysis(df_raw):
    print("Shape (rows, columns): ", df_raw.shape)

    print("\nHead:")
    print(df_raw.head())

    print("\nInfo:")
    print(df_raw.info())

    print("\nDescribe:")
    print(df_raw.describe())

    print("\nMissing Values per Column:")
    print(df_raw.isna().sum())

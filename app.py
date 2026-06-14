import streamlit as st
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_icon=":==")
full_df = df = pd.read_csv('full_df.csv')

st.sidebar.title("Filters")

available_types = ['Objective', 'Subjective', 'Mixed']
selected_types = st.sidebar.multiselect("News Types to include", options=available_types, default=available_types)

min_pol, max_pol = float(full_df['polarity_mean'].min()), float(full_df['polarity_mean'].max())
pol_range = st.sidebar.slider("Polarity Mean Range", min_value=min_pol, max_value=max_pol, value=(min_pol, max_pol),
                              step=0.01)

filtered_df = full_df.copy()
if 'news_type' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['news_type'].isin(selected_types)]
if pol_range and 'polarity_mean' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['polarity_mean'] >= pol_range[0]) &
        (filtered_df['polarity_mean'] <= pol_range[1])
        ]

st.title("Crypto News Sentiment and Market Reaction Analysis")
st.markdown(
    """**Project:** See how news sentiment affects cryptocurrency market returns, more specifically on **altcoins** and **blue-chip** coins""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data Overview and Quality",
    "2. Descriptive Statistics",
    "3. Visualizations and Overview",
    "4. Hypothesis Testing",
    "5. Discussion and Insights"
])

with tab1:
    st.header("Dataset Description and Cleanup Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {len(full_df)}, {full_df.shape[1]}")
        st.write("**Key numerical fields:** `polarity_mean`, `subjectivity_mean`, `Altcoin_Index`, `Bluechip_Index`")

        st.markdown("""
        **Data Sources:**
        - News: Kaggle CryptoNews dataset (headlines, polarity, subjectivity)
        - Market: Yahoo Finance (daily prices for BTC, ETH + altcoins)
        - **Final dataset** was created by daily aggregation of news sentiment + market returns.
        """)

        st.subheader("Data Quality Checks")
        st.write(f"Total nan values: **{full_df.isnull().sum().sum():,}**")

        st.markdown("""
        **Cleanup steps:**
        - Parsed `sentiment` string -> extracted `polarity`, `subjectivity`, `class`
        - `pd.to_datetime` for consistent date format
        - Dropped rows with NaNs (polarity/subjectivity)
        - IQR-based outlier handling for extreme returns
        - Created `news_type` column (Objective / Subjective / Mixed based on subjectivity)
        """)

    with col2:
        st.subheader("Sample of Filtered Data")
        st.dataframe(filtered_df.head(8))

with tab2:
    st.header("Descriptive Statistics for Key Numerical Fields")
    st.markdown("polarity_mean, subjectivity_mean, Altcoin_Index, Bluechip_Index")

    stats_df = filtered_df[['polarity_mean', 'subjectivity_mean', 'Altcoin_Index', 'Bluechip_Index']].describe().T[
        ['mean', '50%', 'std', 'min', 'max']].round(4)
    stats_df.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
    st.dataframe(stats_df)

    st.subheader("News Type Distribution")
    type_counts = filtered_df['news_type'].value_counts()
    st.bar_chart(type_counts)

with tab3:
    st.header("Plots for Numerical Fields and Detailed Comparisons")
    st.subheader("Market Reactions and Correlations")

    c1, c2 = st.columns(2)

    with c1:
        df_grouped = filtered_df.groupby(['relative_sentiment', 'subject'])['BTC_Return'].agg(
            ['mean', 'std']).reset_index()

        fig_a = px.bar(
            df_grouped,
            x='relative_sentiment',
            y='mean',
            error_y='std',
            color='subject',
            barmode='group',
            category_orders={"relative_sentiment": ["Negative", "Neutral", "Positive"]},  # <- THE FIX
            title='BTC returns by relative sentiment and subject of news',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        st.plotly_chart(fig_a)

    with c2:
        fig_b = px.scatter(
            filtered_df,
            x='Bluechip_Index',
            y='Altcoin_Index',
            color='relative_sentiment',
            color_discrete_map={'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Positive': '#2ecc71'},
            opacity=0.7,
            title='Bluechip vs altcoin market reaction to news',
        )

        min_val = min(filtered_df['Bluechip_Index'].min(), filtered_df['Altcoin_Index'].min())
        max_val = max(filtered_df['Bluechip_Index'].max(), filtered_df['Altcoin_Index'].max())
        fig_b.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="black", dash="dot", width=1.5)
        )
        st.plotly_chart(fig_b)

    c3, c4 = st.columns(2)

    with c3:
        fig_c = px.violin(
            filtered_df,
            x='subject',
            y='Bluechip_Index',
            color='news_type',
            box=True,
            title='Bluechip returns distributed by subject and news type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_c)

    with c4:
        num_cols = filtered_df[
            ['polarity_mean', 'subjectivity_mean', 'article_count', 'Altcoin_Index', 'Bluechip_Index', 'BTC_Return',
             'ETH_Return']]
        corr = num_cols.corr()

        fig_d = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Correlation heatmap'
        )
        st.plotly_chart(fig_d)

    st.divider()
    st.subheader("Distributions, Timelines, and Subjectivity")

    c5, c6 = st.columns(2)

    with c5:
        fig_e = px.histogram(
            filtered_df,
            x='polarity_mean',
            nbins=50,
            title='Distribution of mean news polarity',
            color_discrete_sequence=['steelblue']
        )
        st.plotly_chart(fig_e)

    with c6:
        fig_f = px.box(
            filtered_df,
            x='subject',
            y='article_count',
            title='Distribution of article count per subject',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_f)

    c7, c8 = st.columns(2)

    with c7:
        fig_g = px.line(
            filtered_df,
            x='date',
            y='Bluechip_Index',
            title='Bluechip Returns',
            color_discrete_sequence=['coral']
        )
        fig_g.update_xaxes(nticks=10)
        st.plotly_chart(fig_g)

    with c8:
        fig_h = px.scatter(
            filtered_df,
            x='subjectivity_mean',
            y='Altcoin_Index',
            color='subject',
            title='Subjectivity mean vs altcoin returns',
            labels={'subjectivity_mean': 'Subjectivity', 'Altcoin_Index': 'Altcoin Index'},
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_h, use_container_width=True)
with tab4:
    st.header("Hypothesis Testing: Altcoin Returns by News Type")
    st.markdown("""
    **Hypothesis 1:** Altcoin market reactions differ significantly depending on news type 
    (Objective or Subjective or Mixed). Altcoins are expected to show stronger reactions to subjective/opinionated news.
    """)

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.subheader("Pairwise Tests (Bonferroni Correction)")
        news_types = ['Objective', 'Subjective', 'Mixed']
        alpha = 0.05
        n_comparisons = len(list(combinations(news_types, 2)))
        bonferroni_alpha = alpha / n_comparisons

        st.write(f"**Bonferroni-adjusted alpha:** `{bonferroni_alpha:.4f}` (for {n_comparisons} tests)")

        results = []
        for n1, n2 in combinations(news_types, 2):
            d1 = filtered_df[filtered_df['news_type'] == n1]['Altcoin_Index'].dropna()
            d2 = filtered_df[filtered_df['news_type'] == n2]['Altcoin_Index'].dropna()
            if len(d1) > 1 and len(d2) > 1:
                t, p = stats.ttest_ind(d1, d2, equal_var=False)
                sig = "SIGNIFICANT" if p < bonferroni_alpha else "not significant"
                results.append({
                    'Comparison': f"{n1} vs {n2}",
                    'n1': len(d1), 'n2': len(d2),
                    't-stat': round(t, 3),
                    'p-value': round(p, 4),
                    'Significant (Bonf)': sig
                })

        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("Not enough data in selected filters for all comparisons.")

    with c2:
        st.subheader("Data Distribution")
        fig, ax = plt.subplots(figsize=(9, 6))

        sns.boxplot(data=filtered_df, x='news_type', y='Altcoin_Index',
                    order=news_types, palette='pastel', showfliers=False, ax=ax)
        sns.stripplot(data=filtered_df, x='news_type', y='Altcoin_Index',
                      order=news_types, color='black', alpha=0.5, jitter=True, size=4, ax=ax)

        ax.set_title('Altcoin Index by News Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('News Type')
        ax.set_ylabel('Altcoin Index')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)

        st.pyplot(fig)

    st.markdown("""
        **Hypothesis 2:** Subjectivity affects polarity strength
        We expect higher subjectivity to be associated with more extreme (positive or negative) polarity.
        """)

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.subheader("Pearson Correlation Test")
        plot_df = filtered_df.copy()
        plot_df['polarity_abs'] = plot_df['polarity_mean'].abs()

        valid = plot_df[['subjectivity_mean', 'polarity_abs']].dropna()
        if len(valid) > 1:
            r, p = stats.pearsonr(valid['subjectivity_mean'], valid['polarity_abs'])
            st.write(f"**Pearson r** between `subjectivity_mean` and absolute value of `polarity_mean`: **{r:.4f}**")
            st.write(f"**p-value**: `{p:.4e}`")

            if p < 0.05:
                if p < 0.05:
                    if r > 0:
                        st.success(
                            "**Significant positive relationship** — higher subjectivity leads to stronger sentiment polarity.")
                    elif r < 0:
                        st.success(
                            "** Significant NEGATIVE relationship** — higher subjectivity is linked to WEAKER emotional intensity.")
            else:
                st.warning("No significant correlation found in the current filtered data.")
        else:
            st.warning("Not enough valid data points after applying filters for the correlation analysis.")

    with c2:
        st.subheader("Scatter Plot: Subjectivity vs Polarity Strength")
        if len(valid) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.scatterplot(
                data=plot_df,
                x='subjectivity_mean',
                y='polarity_abs',
                alpha=0.55,
                color='#3498db',
                s=45,
                ax=ax,
                edgecolor='none',
                label='Daily observations'
            )

            sns.regplot(
                data=plot_df,
                x='subjectivity_mean',
                y='polarity_abs',
                scatter=False,
                color='#e74c3c',
                line_kws={'linewidth': 3, 'label': f'Linear fit (r = {r:.3f})'},
                ax=ax
            )

            ax.set_xlabel('Subjectivity Mean')
            ax.set_ylabel('Absolute Polarity Mean')
            ax.set_title('Does Higher Subjectivity Lead to Stronger Sentiment?', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.spines[['top', 'right']].set_visible(False)

            st.pyplot(fig)
        else:
            st.info("Insufficient data to render the scatter plot under current filters.")

with tab5:
    st.header("Discussion and key insights")
    st.markdown("""
    **What we did and why:**
    - **Data Cleanup** Parsed sentiment strings and standardized dates/types. This is essential because raw data had dicts stored as strings.
    - **Descriptive Stats & Plots:** Showed central tendency and spread of sentiment and market reaction metrics. Used multiple plot types (hist, scatter, box+strip, bar) to give complete picture of distributions and relationships.
    - **Hypothesis Testing:** t-tests with Bonferroni correction. The box+strip plot visualizes both central tendency and individual data points.
    **Limitations and Next Steps:**
    - Daily aggregation may miss intra-day reactions.
    """)
    csv = filtered_df.to_csv()
    st.download_button(label="Download full_df.csv", data=csv, file_name="full_df.csv")

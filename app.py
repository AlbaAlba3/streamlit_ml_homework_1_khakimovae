import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


st.title("HW ML #1 Khakimovae")
st.subheader("Анализ данных о ценах на автомобили")

@st.cache_data
def load_datasets():
    try:
        df_train = pd.read_csv("data/dataset_train.csv")
        df_test = pd.read_csv("data/dataset_test.csv")
        return df_train, df_test
    except FileNotFoundError as e:
        st.error(f"Файл не найден: {e}")
        return None, None
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None, None

df_train, df_test = load_datasets()

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

artifacts = load_model()

with st.expander("Графики", expanded=False):
    st.subheader("Первые строки train:")
    st.dataframe(df_train.head())
    st.subheader("Первые строки test")
    st.dataframe(df_test.head())

    st.subheader("Попарные распределения и корреляции")

    numeric_cols = ["max_power", "engine", "seats", "mileage", "km_driven", "year", "selling_price"]
    numeric_cols_train = [col for col in numeric_cols if col in df_train.columns]
    numeric_cols_test = [col for col in numeric_cols if col in df_test.columns]

    st.write("**Попарные распределения (train)**:")
    if len(numeric_cols_train) > 1:
        pair_grid = sns.pairplot(
            data=df_train[numeric_cols_train],
            height=2.0,
            aspect=1.0,
            diag_kind="kde",
            plot_kws={"s": 15, "alpha": 0.7}
        )
        pair_grid.fig.suptitle("Pairplot: Train Dataset", y=1.02, fontsize=16)
        st.pyplot(pair_grid.fig)
        plt.close(pair_grid.fig)
    else:
        st.warning("Недостаточно числовых колонок для pairplot в train.")

    st.write("**Попарные распределения (test)**:")
    if len(numeric_cols_test) > 1:
        pair_grid = sns.pairplot(
            data=df_test[numeric_cols_test],
            height=2.0,
            aspect=1.0,
            diag_kind="kde",
            plot_kws={"s": 15, "alpha": 0.7}
        )
        pair_grid.fig.suptitle("Pairplot: Test Dataset", y=1.02, fontsize=16)
        st.pyplot(pair_grid.fig)
        plt.close(pair_grid.fig)
    else:
        st.warning("Недостаточно числовых колонок для pairplot в test.")

    st.write("**Тепловая карта корреляций (train)**:")
    correlation_train = df_train[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    heatmap_train = sns.heatmap(
        correlation_train,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={'shrink': 0.8}
    )
    st.pyplot(heatmap_train.figure)
    plt.close()

    st.write("**Тепловая карта корреляций (test)**:")
    correlation_test = df_test[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    heatmap_test = sns.heatmap(
        correlation_test,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={'shrink': 0.8}
    )
    st.pyplot(heatmap_test.figure)
    plt.close()

    st.subheader("Средние цены по категориальным признакам")

    avg_price = df_train.groupby('owner')['selling_price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price.plot(kind='bar', ax=ax)
    ax.set_title('Средняя цена по категориям владельцев')
    ax.set_xlabel('Категория владельца')
    ax.set_ylabel('Средняя цена (руб.)')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg_price = df_train.groupby('fuel')['selling_price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price.plot(kind='bar', ax=ax)
    ax.set_title('Средняя цена по категориям топлива')
    ax.set_xlabel('Категория топлива')
    ax.set_ylabel('Средняя цена (руб.)')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg_price = df_train.groupby('transmission')['selling_price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price.plot(kind='bar', ax=ax)
    ax.set_title('Средняя цена по типам трансмиссии')
    ax.set_xlabel('Тип трансмиссии')
    ax.set_ylabel('Средняя цена (руб.)')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg_price = df_train.groupby('seller_type')['selling_price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price.plot(kind='bar', ax=ax)
    ax.set_title('Средняя цена по типам продавцов')
    ax.set_xlabel('Тип продавца')
    ax.set_ylabel('Средняя цена (руб.)')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

#ИСПОЛЬЗОВАЛ ГУГЛ И LLM для оформления красивых инпутов и валидаций
with st.expander("Прогноз цены автомобиля", expanded=False):
    st.write("Введите характеристики автомобиля:")

    with st.form("car_prediction_form"):
        st.subheader("Основные характеристики")

        col1, col2 = st.columns(2)

        with col1:
            name_index = artifacts['categorical_cols'].index('name')
            available_brands = list(artifacts['encoder'].categories_[name_index])
            available_brands = sorted([str(brand).strip().lower() for brand in available_brands])

            brand = st.selectbox(
                "Марка автомобиля *",
                options=available_brands,
                help="Выберите марку автомобиля из списка"
            )

            model_name = st.text_input(
                "Модель автомобиля",
                help="Введите модель автомобиля"
            )

            year = st.number_input(
                "Год выпуска",
                min_value=1990.0,
                max_value=2025.0,
                value=2018.0,
                step=1.0,
                format="%.0f",
                help="Год выпуска автомобиля"
            )

            km_driven = st.number_input(
                "Пробег (км)",
                min_value=0.0,
                max_value=500000.0,
                value=30000.0,
                step=1000.0,
                format="%.0f",
                help="Общий пробег автомобиля в километрах"
            )

            mileage = st.number_input(
                "Расход топлива",
                min_value=5.0,
                max_value=100.0,
                value=14.5,
                step=0.1,
                format="%.1f",
                help="Средний расход топлива"
            )

        with col2:
            engine = st.number_input(
                "Объем двигателя (см^3)",
                min_value=500.0,
                max_value=5000.0,
                value=1984.0,
                step=100.0,
                format="%.0f",
                help="Объем двигателя"
            )

            max_power = st.number_input(
                "Мощность (л.с.)",
                min_value=30.0,
                max_value=500.0,
                value=150.0,
                step=5.0,
                format="%.1f",
                help="Мощность двигателя в лошадиных силах"
            )

            fuel = st.selectbox(
                "Тип топлива *",
                options=['Petrol', 'Diesel', 'CNG', 'LPG'],
                help="Тип используемого топлива"
            )

            seller_type = st.selectbox(
                "Тип продавца *",
                options=['Dealer', 'Individual', 'Trustmark Dealer'],
                help="Тип продавца автомобиля"
            )

            transmission = st.selectbox(
                "Коробка передач *",
                options=['Automatic', 'Manual'],
                help="Тип коробки передач"
            )

            owner = st.selectbox(
                "Количество владельцев *",
                options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                help="Сколько было владельцев у автомобиля"
            )

            seats = st.selectbox(
                "Количество мест *",
                options=[2, 4, 5, 6, 7, 8, 9, 10, 14],
                help="Количество посадочных мест",
                format_func=lambda x: str(x)
            )

            submitted = st.form_submit_button("Рассчитать цену")

        if submitted and artifacts is not None:
            if not brand:
                st.error("Пожалуйста, выберите марку автомобиля")
            else:
                try:
                    input_data = pd.DataFrame([{
                        'year': float(year),
                        'km_driven': float(km_driven),
                        'mileage': float(mileage),
                        'engine': float(engine),
                        'max_power': float(max_power),
                        'name': str(brand).strip().lower(),
                        'fuel': str(fuel),
                        'seller_type': str(seller_type),
                        'transmission': str(transmission),
                        'owner': str(owner),
                        'seats': str(seats)
                    }])

                    with st.spinner("Выполняется расчёт цены..."):
                        try:
                            categorical_cols = artifacts['categorical_cols']
                            input_categorical = input_data[categorical_cols].copy()

                            for col in categorical_cols:
                                input_categorical[col] = input_categorical[col].astype(str)

                            X_encoded = artifacts['encoder'].transform(input_categorical)
                            encoded_features = artifacts['encoder'].get_feature_names_out(categorical_cols)
                            encoded_df = pd.DataFrame(X_encoded, columns=encoded_features)

                            numerical_cols = artifacts['numerical_cols']
                            numerical_data = input_data[numerical_cols].copy()

                            X_processed = pd.concat([numerical_data, encoded_df], axis=1)

                            final_feature_order = artifacts['final_feature_order']
                            for col in final_feature_order:
                                if col not in X_processed.columns:
                                    X_processed[col] = 0

                            X_processed = X_processed[final_feature_order]

                            X_scaled = artifacts['scaler'].transform(X_processed)

                            prediction = artifacts['model'].predict(X_scaled)[0]

                            st.success(f"Предсказанная цена: **{prediction:,.0f} ₽**")

                        except Exception as e:
                            st.error(f"Ошибка при расчёте цены: {e}")
                            st.info("Попробуйте изменить параметры или проверьте корректность введённых данных")

                except Exception as e:
                    st.error(f"Ошибка при обработке данных: {e}")
        elif submitted and artifacts is None:
            st.error("Модель не загружена. Невозможно выполнить расчёт.")


#ИСПОЛЬЗОВАЛ ГУГЛ И LLM для валидаци из csv файла + по использованию виджетов консультировался
with st.expander("Прогноз цены по CSV‑файлу - если вашего авто не было в форме", expanded=False):
    st.write("Загрузите CSV‑файл с данными об автомобилях для массового предсказания цен.")
    st.caption(
        "Ожидаемый формат: столбцы `name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats`")

    uploaded_file = st.file_uploader(
        "Выберите CSV‑файл",
        type=["csv"],
        help="Файл должен содержать указанные столбцы. selling_price будет предсказан моделью."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required_cols = [
                'name', 'year', 'km_driven', 'fuel', 'seller_type',
                'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(
                    f"Не хватает столбцов: {', '.join(missing_cols)}. Ожидаемые столбцы: {', '.join(required_cols)}")
                st.stop()

            st.info(f"Загружено {len(df)} записей. Выполняется валидация данных...")

            errors = []
            for idx, row in df.iterrows():
                for col in ['year', 'km_driven', 'mileage', 'engine', 'max_power']:
                    if pd.isna(row[col]):
                        errors.append(f"Строка {idx + 2}: отсутствует значение в столбце '{col}'")
                    elif not isinstance(row[col], (int, float)) or row[col] <= 0:
                        errors.append(f"Строка {idx + 2}: некорректное значение '{row[col]}' в столбце '{col}'")

                if pd.isna(row['seats']):
                    errors.append(f"Строка {idx + 2}: отсутствует значение в столбце 'seats'")
                elif not isinstance(row['seats'], (int, float)) or int(row['seats']) < 2:
                    errors.append(
                        f"Строка {idx + 2}: некорректное значение '{row['seats']}' в столбце 'seats' (должно быть ≥2)")

                name_index = artifacts['categorical_cols'].index('name')
                available_brands = [str(brand).strip().lower() for brand in
                                    artifacts['encoder'].categories_[name_index]]
                if str(row['name']).strip().lower() not in available_brands:
                    errors.append(f"Строка {idx + 2}: марка '{row['name']}' не найдена в обучающих данных")

                if str(row['fuel']) not in ['Petrol', 'Diesel', 'CNG', 'LPG']:
                    errors.append(f"Строка {idx + 2}: тип топлива '{row['fuel']}' не поддерживается")

                if str(row['seller_type']) not in ['Dealer', 'Individual', 'Trustmark Dealer']:
                    errors.append(f"Строка {idx + 2}: тип продавца '{row['seller_type']}' не поддерживается")

                if str(row['transmission']) not in ['Automatic', 'Manual']:
                    errors.append(f"Строка {idx + 2}: коробка передач '{row['transmission']}' не поддерживается")

                if str(row['owner']) not in ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner',
                                             'Test Drive Car']:
                    errors.append(f"Строка {idx + 2}: количество владельцев '{row['owner']}' не поддерживается")

            if errors:
                st.error("Найденные ошибки в данных:")
                for err in errors:
                    st.write(err)
                st.stop()

            st.success("Данные успешно валидированы! Выполняется предсказание...")

            input_data = df[required_cols].copy()
            input_data['name'] = input_data['name'].astype(str).str.strip().str.lower()

            categorical_cols = artifacts['categorical_cols']
            input_categorical = input_data[categorical_cols].copy()
            for col in categorical_cols:
                input_categorical[col] = input_categorical[col].astype(str)

            X_encoded = artifacts['encoder'].transform(input_categorical)
            encoded_features = artifacts['encoder'].get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(X_encoded, columns=encoded_features)

            numerical_cols = artifacts['numerical_cols']
            numerical_data = input_data[numerical_cols].copy()

            X_processed = pd.concat([numerical_data, encoded_df], axis=1)
            final_feature_order = artifacts['final_feature_order']

            for col in final_feature_order:
                if col not in X_processed.columns:
                    X_processed[col] = 0

            X_processed = X_processed[final_feature_order]

            X_scaled = artifacts['scaler'].transform(X_processed)
            predictions = artifacts['model'].predict(X_scaled)

            df['predicted_price'] = predictions

            st.subheader("Результаты предсказания")
            predicted_price = predictions[0]
            st.success(f"Предсказанная цена: **{predicted_price:,.0f} ₽**")


        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")
            st.code(str(e))


with st.expander("Визуализация коэффициентов модели", expanded=False):
    st.subheader("Веса признаков в модели")

    if artifacts is not None:
        feature_names = artifacts['final_feature_order']
        coeffs = artifacts['model'].coef_

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coeffs
        }).sort_values('feature', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 12))
        ax.barh(coef_df['feature'], coef_df['coefficient'])

        ax.set_xlabel("Значение коэффициента (вес признака)")
        ax.set_ylabel("Признак")
        ax.set_title("Веса признаков в модели")

        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Модель не загружена. Невозможно отобразить коэффициенты.")
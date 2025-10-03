from .portfolios import Portfolios


class PortfolioService:
    def __init__(self, portfolios: Portfolios):
        self.portfolios = portfolios

    def get_all_customer_ids(self) -> list[int]:
        return self.portfolios.df_out["customer_id"].unique().tolist()

    def get_port_id_from_customer_id(self, customer_id: int) -> int:
        result = self.portfolios.port_id_mapping[self.portfolios.port_id_mapping['customer_id'] == customer_id]['port_id'].values

        if len(result) == 0:
            raise ValueError(f"No portfolio found for customer_id={customer_id}")
        if len(result) > 1:
            raise ValueError(f"More than one portfolio found for customer_id={customer_id}")
        return int(result[0])

    def get_client_portfolio(self, customer_id: int) -> Portfolios:
        port_id = self.get_port_id_from_customer_id(customer_id)


        # slice the data
        df_out_client = self.portfolios.df_out[self.portfolios.df_out["port_id"] == port_id]
        df_style_client = self.portfolios.df_style[self.portfolios.df_style["port_id"] == port_id]
        port_id_mapping = self.portfolios.port_id_mapping[self.portfolios.port_id_mapping["port_id"] == port_id]
        port_ids = self.portfolios.port_ids[self.portfolios.port_ids == port_id]

        # create a temporary Portfolios instance just for this client
        client_portfolio = Portfolios()
        client_portfolio.set_ref_tables({
            "product_mapping": self.portfolios.product_mapping,
            "product_underlying": self.portfolios.product_underlying
        })
        client_portfolio.set_portfolio(df_out_client, df_style_client, port_ids, port_id_mapping)
        return client_portfolio

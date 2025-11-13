"""Configuration primitives for the finance AI toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AnomalyConfig:
	"""Hyper-parameters used by the anomaly detector."""

	contamination: float = 0.05
	random_state: int = 42
	feature_columns: Sequence[str] = (
		"amount",
		"abs_amount",
		"rolling_7d_spend",
		"rolling_30d_spend",
		"day_of_week",
		"is_weekend",
		"hour",
	)


@dataclass(slots=True)
class QualityConfig:
	"""Settings for the monthly data-quality assessor."""

	nu: float = 0.05
	gamma: float | Literal["scale", "auto"] = "scale"
	score_threshold: float = -0.05
	min_months: int = 4


@dataclass(slots=True)
class ForecastConfig:
	"""Settings for the expense forecaster."""

	horizon_months: int = 6
	seasonal_periods: int = 12
	damped_trend: bool = True


@dataclass(slots=True)
class FinanceAIConfig:
	"""Container for reusable configuration across the project."""

	currency: str = "BRL"
	date_column: str = "date"
	description_column: str = "title"
	amount_column: str = "amount"
	raw_data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
	processed_dirname: str = "processed"
	income_keywords: Sequence[str] = (
		"pagamento recebido",
		"recebido",
		"transferencia",
		"salary",
		"deposito",
		"estorno",
	)
	refund_keywords: Sequence[str] = (
		"estorno",
		"chargeback",
		"reversal",
		"refund",
	)
	subscription_keywords: Sequence[str] = (
		"spotify",
		"netflix",
		"microsoft",
		"apple",
		"amazon",
		"google",
		"openai",
		"nuv",
	)
	category_keywords: Dict[str, Sequence[str]] = field(
		default_factory=lambda: {
			"Alimentacao": (
				"yakide",
				"dominos",
				"subway",
				"pizz",
				"pizzaria",
				"rest",
				"restaurante",
				"ifood",
				"ubereats",
				"rapi",
				"padaria",
				"cafe",
				"cafeteria",
				"bistro",
				"lanch",
				"lanche",
				"mercado",
				"quiosque",
				"sushi",
				"burger",
				"fast food",
				"zona sul",
				"mate",
			),
			"Supermercado": (
				"supermercado",
				"hiper",
				"mercadopago",
				"mercado livre",
				"bahamas",
				"pao de acucar",
				"atacadao",
				"assai",
				"carrefour",
				"guanabara",
				"ultra",
				"casa do biscoito",
			),
			"Transporte": (
				"uber",
				"99",
				"cabify",
				"nupay",
				"clickbus",
				"rodoviaria",
				"passagem",
				"passagens",
				"bilhete",
				"metro",
				"bus",
				"trip",
				"pedagio",
				"estacionamento",
				"locadora",
			),
			"Combustivel": (
				"posto",
				"ipiranga",
				"shell",
				"petrobras",
				"combust",
				"gasolina",
				"diesel",
				"alcool",
			),
			"Habitacao": (
				"aluguel",
				"condominio",
				"imobili",
				"energia",
				"luz",
				"agua",
				"gas",
				"materiais",
				"construcao",
				"manutencao",
				"ferramenta",
			),
			"Telecom": (
				"internet",
				"banda larga",
				"fibra",
				"telefon",
				"celular",
				"claro",
				"vivo",
				"tim",
				"oi",
				"net",
				"sky",
			),
			"Saude": (
				"farm",
				"drog",
				"clin",
				"med",
				"hospital",
				"laboratorio",
				"odonto",
				"dental",
			),
			"Educacao": (
				"curso",
				"livro",
				"escola",
				"faculdade",
				"univers",
				"idioma",
				"catarse",
			),
			"Servicos": (
				"cassiano",
				"vivian",
				"serv",
				"consult",
				"pagamento",
				"okto",
				"assessoria",
				"assistencia",
				"contab",
				"agencia",
				"manutencao",
			),
			"Entretenimento": (
				"cinema",
				"teatro",
				"show",
				"evento",
				"ingresso",
				"netflix",
				"spotify",
				"prime video",
				"disney",
				"hbo",
				"game",
				"playstation",
				"xbox",
				"steam",
				"cinemark",
			),
			"Lazer": (
				"jfk",
				"taverna",
				"pub",
				"bar",
				"parque",
				"club",
				"piscina",
				"spa",
				"esporte",
			),
			"Compras": (
				"shopping",
				"loja",
				"varejo",
				"magalu",
				"americanas",
				"casas bahia",
				"shopee",
				"shein",
				"centauro",
				"fast shop",
				"decathlon",
				"riachuelo",
			),
			"Beleza": (
				"salon",
				"beleza",
				"cosmet",
				"perfum",
				"sephora",
				"barbearia",
				"estetica",
			),
			"Pets": (
				"petz",
				"petlove",
				"pet shop",
				"agropecu",
				"zoon",
			),
			"Transferencias": (
				"pix",
				"transferencia",
				"transfer",
				"ted",
				"doc",
				"envio",
				"enviar",
				"cash out",
				"picpay",
				"wise",
				"remessa",
			),
			"Investimentos": (
				"invest",
				"tesouro",
				"bolsa",
				"acoes",
				"fundos",
				"cdb",
				"lci",
				"lca",
				"xp",
				"rico",
				"modal",
				"corretora",
			),
			"Tecnologia": (
				"microsoft",
				"google",
				"apple",
				"amazon digital",
				"openai",
				"hardware",
				"software",
				"eletron",
				"gad",
			),
			"Financeiro": (
				"juros",
				"encerramento",
				"imposto",
				"iof",
				"tarifa",
				"taxa",
				"anuidade",
				"banco",
			),
			"ImpostosETaxas": (
				"iptu",
				"ipva",
				"darf",
				"licenc",
			),
			"Seguros": (
				"seguro",
				"porto seguro",
				"bradesco seguros",
				"sulamerica",
				"mapfre",
			),
			"Viagem": (
				"airbnb",
				"hotel",
				"ticket",
				"passagens",
				"viagem",
				"booking",
				"decolar",
				"maxmilhas",
			),
			"Outros": (),
		}
	)
	anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
	quality: QualityConfig = field(default_factory=QualityConfig)
	forecast: ForecastConfig = field(default_factory=ForecastConfig)

	@property
	def processed_data_dir(self) -> Path:
		return self.raw_data_dir / self.processed_dirname

	def ensure_directories(self) -> None:
		self.raw_data_dir.mkdir(parents=True, exist_ok=True)
		self.processed_data_dir.mkdir(parents=True, exist_ok=True)

	def category_for_description(self, description: str) -> str:
		description_lower = description.lower()
		for category, keywords in self.category_keywords.items():
			if any(keyword in description_lower for keyword in keywords):
				return category
		return "Outros"

	def is_income(self, description: str, amount: float) -> bool:
		if amount < 0:
			return True
		description_lower = description.lower()
		return any(keyword in description_lower for keyword in self.income_keywords)

	def is_refund(self, description: str, amount: float) -> bool:
		if amount < 0:
			return True
		description_lower = description.lower()
		return any(keyword in description_lower for keyword in self.refund_keywords)

	def is_subscription(self, description: str) -> bool:
		description_lower = description.lower()
		return any(keyword in description_lower for keyword in self.subscription_keywords)

	def iter_all_keywords(self) -> Iterable[str]:
		for keywords in self.category_keywords.values():
			for keyword in keywords:
				yield keyword


def get_default_config() -> FinanceAIConfig:
	config = FinanceAIConfig()
	config.ensure_directories()
	return config

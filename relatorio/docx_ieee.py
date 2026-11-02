"""
docx_ieee.py
------------
Funções auxiliares para preencher o modelo IEEE (conference-template-a4.docx)
com python-docx, preservando a formatação do modelo.

Este arquivo NÃO faz parte das implementações avaliadas (modelos/experimentos);
serve apenas para montar o relatório.
"""

import copy
import re

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches

LARGURA_COLUNA_TWIPS = 4931          # largura da coluna do modelo (2 colunas)
LARGURA_PAGINA_TWIPS = 10210         # 11910 - 2 x 850 (margens do modelo)
LARGURA_COLUNA_POL = 3.32
LARGURA_PAGINA_POL = 6.95

# Ordem dos filhos de <w:pPr> definida pelo schema WordprocessingML (CT_PPr).
ORDEM_PPR = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
             "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
             "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
             "sectPr", "pPrChange"]
IDIOMA = '<w:lang w:val="pt-BR" w:eastAsia="en-US" w:bidi="ar-SA"/>'


def _texto(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def _esc(texto):
    return (texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# =============================================================================
# Marcação simples de texto: **negrito**, *itálico*, _{subscrito}, ^{sobrescrito}
# =============================================================================
_PADRAO = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def _runs_xml(texto, tamanho=20, italico_base=False, negrito_base=False, small_caps=False):
    partes = []
    for pedaco in _PADRAO.split(texto):
        if not pedaco:
            continue
        b, i, va = negrito_base, italico_base, None
        if pedaco.startswith("**") and pedaco.endswith("**"):
            pedaco, b = pedaco[2:-2], True
        elif pedaco.startswith("*") and pedaco.endswith("*") and len(pedaco) > 1:
            pedaco, i = pedaco[1:-1], not italico_base
        elif pedaco.startswith("_{"):
            pedaco, va = pedaco[2:-1], "subscript"
        elif pedaco.startswith("^{"):
            pedaco, va = pedaco[2:-1], "superscript"
        rpr = ""
        if b:
            rpr += "<w:b/>"
        if i:
            rpr += "<w:i/>"
        if small_caps:
            rpr += "<w:smallCaps/>"
        if va:
            rpr += f'<w:vertAlign w:val="{va}"/>'
        rpr += f'<w:sz w:val="{tamanho}"/><w:szCs w:val="{tamanho}"/>{IDIOMA}'
        partes.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{_esc(pedaco)}</w:t></w:r>')
    return "".join(partes)


# =============================================================================
# Equações (Office Math / OMML)
# =============================================================================
class Eq:
    """Construtor mínimo de OMML. Cada método devolve um trecho XML."""

    @staticmethod
    def r(texto, estilo=None):
        """Trecho de texto matemático. estilo: None (itálico), "p" (normal) ou "b" (negrito)."""
        sty = f'<m:rPr><m:sty m:val="{estilo}"/></m:rPr>' if estilo else ""
        return (f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
                f'<m:t xml:space="preserve">{_esc(texto)}</m:t></m:r>')

    @staticmethod
    def t(texto):  # texto "normal" (não itálico), ex.: nomes de funções e números
        return Eq.r(texto, "p")

    @staticmethod
    def b(texto):  # vetores e matrizes em negrito
        return Eq.r(texto, "b")

    @staticmethod
    def sub(base, indice):
        return f"<m:sSub><m:e>{base}</m:e><m:sub>{indice}</m:sub></m:sSub>"

    @staticmethod
    def sup(base, expoente):
        return f"<m:sSup><m:e>{base}</m:e><m:sup>{expoente}</m:sup></m:sSup>"

    @staticmethod
    def subsup(base, indice, expoente):
        return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{indice}</m:sub><m:sup>{expoente}</m:sup></m:sSubSup>"

    @staticmethod
    def frac(num, den):
        return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"

    @staticmethod
    def soma(inferior, superior, corpo):
        return ('<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>'
                f"<m:sub>{inferior}</m:sub><m:sup>{superior}</m:sup><m:e>{corpo}</m:e></m:nary>")

    @staticmethod
    def par(conteudo, abre="(", fecha=")"):
        return (f'<m:d><m:dPr><m:begChr m:val="{_esc(abre)}"/><m:endChr m:val="{_esc(fecha)}"/></m:dPr>'
                f"<m:e>{conteudo}</m:e></m:d>")

    @staticmethod
    def raiz(conteudo):
        return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{conteudo}</m:e></m:rad>'

    @staticmethod
    def acento(base, caractere="̃"):
        return f'<m:acc><m:accPr><m:chr m:val="{caractere}"/></m:accPr><m:e>{base}</m:e></m:acc>'

    @staticmethod
    def barra(base):
        return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{base}</m:e></m:bar>'

    @staticmethod
    def casos(*linhas):
        """Chave à esquerda com uma linha por caso (ex.: função degrau)."""
        eqarr = "".join(f"<m:e>{l}</m:e>" for l in linhas)
        return Eq.par(f"<m:eqArr>{eqarr}</m:eqArr>", "{", "")


# =============================================================================
# Documento
# =============================================================================
class RelatorioIEEE:
    """Abre o modelo, guarda cópias dos parágrafos-modelo e limpa o conteúdo
    de exemplo, mantendo título, bloco de autores e configuração de página."""

    def __init__(self, caminho_modelo):
        self.doc = Document(caminho_modelo)
        self.body = self.doc.element.body
        elementos = list(self.body.iterchildren())
        self.sect_final = elementos[-1]

        def achar(trecho):
            for el in elementos:
                if el.tag == qn("w:p") and trecho in _texto(el):
                    return el
            raise KeyError(trecho)

        # Modelos de formatação (cópias profundas).
        self.m_titulo = elementos[0]
        self.m_resumo = copy.deepcopy(achar("This electronic document"))
        self.m_palavras = copy.deepcopy(achar("component, formatting"))
        self.m_corpo = copy.deepcopy(achar("This template, modified"))
        self.m_legenda_fig = copy.deepcopy(achar("Example of a figure caption"))
        self.m_legenda_tab = copy.deepcopy(achar("Table Type Styles"))
        self.m_referencia = copy.deepcopy(achar("G. Eason"))
        # Seção de 2 colunas do corpo do artigo (fica no pPr de um parágrafo).
        self.sect_duas_colunas = None
        for el in elementos:
            sect = el.find(qn("w:pPr") + "/" + qn("w:sectPr")) if el.tag == qn("w:p") else None
            if sect is not None and sect.find(qn("w:cols")) is not None and \
                    sect.find(qn("w:cols")).get(qn("w:num")) == "2":
                self.sect_duas_colunas = copy.deepcopy(sect)
                break
        self._cont_fig = 0
        self._cont_tab = 0
        self._cont_eq = 0

        # Autores: o modelo tem 2 linhas x 3 colunas; ficamos só com a 1ª linha.
        self._autores = []
        for rotulo in ("1st Given", "2nd Given", "3rd Given"):
            inicio = elementos.index(achar(rotulo))
            self._autores.append(elementos[inicio:inicio + 4])
        for rotulo in ("4th Given", "5th Given", "6th Given"):
            inicio = elementos.index(achar(rotulo))
            for el in elementos[inicio - 1:inicio + 4]:  # inclui o parágrafo vazio separador
                self.body.remove(el)
        # Remove a nota de subtítulo e todo o conteúdo de exemplo após os autores.
        self.body.remove(achar("Sub-titles are not captured"))
        inicio_exemplo = elementos.index(achar("This electronic document"))
        for el in elementos[inicio_exemplo:-1]:
            if el.getparent() is not None:
                self.body.remove(el)

    # ------------------------------------------------------------------ util
    def _anexar(self, el):
        self.sect_final.addprevious(el)
        return el

    def _paragrafo(self, modelo, runs_xml):
        p = copy.deepcopy(modelo)
        for filho in list(p):
            if filho.tag != qn("w:pPr"):
                p.remove(filho)
        for r in parse_xml(f"<w:root {nsdecls('w')}>{runs_xml}</w:root>"):
            p.append(r)
        return p

    @staticmethod
    def _ajustar_ppr(p, xml_extra=None, remover=()):
        """Remove/acrescenta propriedades de parágrafo e reordena os filhos de w:pPr
        na sequência exigida pelo schema do OOXML (o Word rejeita outra ordem)."""
        ppr = p.find(qn("w:pPr"))
        for tag in remover:
            for el in ppr.findall(qn(tag)):
                ppr.remove(el)
        if xml_extra:
            for el in parse_xml(f"<w:root {nsdecls('w')}>{xml_extra}</w:root>"):
                for repetido in ppr.findall(el.tag):  # evita elementos duplicados
                    ppr.remove(repetido)
                ppr.append(el)
        filhos = list(ppr)
        filhos.sort(key=lambda el: ORDEM_PPR.index(el.tag.split("}")[1])
                    if el.tag.split("}")[1] in ORDEM_PPR else len(ORDEM_PPR))
        for el in filhos:
            ppr.remove(el)
            ppr.append(el)
        return p

    # --------------------------------------------------------- cabeçalho
    def titulo(self, texto):
        for filho in list(self.m_titulo):
            if filho.tag != qn("w:pPr"):
                self.m_titulo.remove(filho)
        for r in parse_xml(f"<w:root {nsdecls('w')}>{_runs_xml(texto, 48)}</w:root>"):
            self.m_titulo.append(r)

    def autores(self, lista):
        """lista: até 3 tuplas (nome, curso, instituição, cidade/país, email)."""
        for bloco, dados in zip(self._autores, lista):
            nome, curso, instituicao, cidade, email = dados
            linhas = [
                _runs_xml(nome, 22) + '<w:r><w:br/></w:r>' + _runs_xml(curso, 18, italico_base=True),
                _runs_xml(instituicao, 18, italico_base=True),
                _runs_xml(cidade, 18),
                _runs_xml(email, 18),
            ]
            for p, runs in zip(bloco, linhas):
                quebra_coluna = p.find(".//" + qn("w:br"))
                tem_quebra = quebra_coluna is not None and quebra_coluna.get(qn("w:type")) == "column"
                for filho in list(p):
                    if filho.tag != qn("w:pPr"):
                        p.remove(filho)
                if tem_quebra:
                    runs = '<w:r><w:br w:type="column"/></w:r>' + runs
                for r in parse_xml(f"<w:root {nsdecls('w')}>{runs}</w:root>"):
                    p.append(r)

    def resumo(self, texto, palavras_chave):
        p = self._paragrafo(self.m_resumo, _runs_xml("Resumo", 18, italico_base=True, negrito_base=True)
                            + _runs_xml("—" + texto, 18, negrito_base=True))
        self._anexar(p)
        p = self._paragrafo(self.m_palavras, _runs_xml("Palavras-chave", 18, True, True)
                            + _runs_xml("—" + palavras_chave, 18, italico_base=True, negrito_base=True))
        self._anexar(p)

    # ---------------------------------------------------------- corpo
    def secao(self, numero_romano, texto):
        """Título de seção (Heading 1 do IEEE): centralizado, versalete."""
        p = self._paragrafo(self.m_corpo, _runs_xml(f"{numero_romano}. {texto}", 20, small_caps=True))
        self._ajustar_ppr(p, '<w:keepNext/><w:spacing w:before="200" w:after="80"/><w:ind w:firstLine="0"/>'
                             '<w:jc w:val="center"/>', remover=("w:spacing", "w:ind", "w:jc"))
        return self._anexar(p)

    def subsecao(self, letra, texto):
        """Heading 2 do IEEE: letra + texto em itálico, alinhado à esquerda."""
        p = self._paragrafo(self.m_corpo, _runs_xml(f"{letra}. {texto}", 20, italico_base=True))
        self._ajustar_ppr(p, '<w:keepNext/><w:spacing w:before="120" w:after="40"/><w:ind w:firstLine="0"/>'
                             '<w:jc w:val="left"/>', remover=("w:spacing", "w:ind", "w:jc"))
        return self._anexar(p)

    def paragrafo(self, texto, titulo_em_linha=None):
        """Parágrafo de corpo. titulo_em_linha gera o Heading 3 do IEEE
        ("1) Título: texto...")."""
        runs = ""
        if titulo_em_linha:
            runs += _runs_xml(titulo_em_linha + ": ", 20, italico_base=True)
        runs += _runs_xml(texto, 20)
        p = self._paragrafo(self.m_corpo, runs)
        self._ajustar_ppr(p, '<w:spacing w:line="228" w:lineRule="auto" w:before="60"/>', remover=("w:spacing",))
        return self._anexar(p)

    def item(self, texto, marcador="•"):
        p = self._paragrafo(self.m_corpo, _runs_xml(f"{marcador}\t", 20) + _runs_xml(texto, 20))
        self._ajustar_ppr(p, '<w:tabs><w:tab w:val="left" w:pos="345"/></w:tabs>'
                             '<w:spacing w:line="228" w:lineRule="auto" w:before="30"/>'
                             '<w:ind w:left="345" w:hanging="288"/>', remover=("w:spacing", "w:ind"))
        return self._anexar(p)

    def equacao(self, conteudo_omml):
        """Equação centralizada com número à direita, como no modelo IEEE."""
        self._cont_eq += 1
        centro = LARGURA_COLUNA_TWIPS // 2
        xml = (f'<w:p {nsdecls("w", "m")}><w:pPr><w:pStyle w:val="BodyText"/>'
               f'<w:tabs><w:tab w:val="center" w:pos="{centro}"/>'
               f'<w:tab w:val="right" w:pos="{LARGURA_COLUNA_TWIPS - 60}"/></w:tabs>'
               '<w:spacing w:before="80" w:after="80"/><w:ind w:left="0" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>'
               '<w:r><w:tab/></w:r>'
               f'<m:oMath>{conteudo_omml}</m:oMath>'
               f'<w:r><w:tab/></w:r><w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>({self._cont_eq})</w:t></w:r></w:p>')
        self._anexar(parse_xml(xml))
        return self._cont_eq

    def proxima_equacao(self):
        return self._cont_eq + 1

    # ------------------------------------------------------- figuras
    def proxima_figura(self):
        return self._cont_fig + 1

    def figura(self, caminho, legenda, largura_pol=LARGURA_COLUNA_POL):
        """Figura na largura da coluna, com legenda no estilo do modelo."""
        self._cont_fig += 1
        p = self.doc.add_paragraph()  # python-docx insere antes do sectPr final
        p.paragraph_format.alignment = 1
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = 0
        p.add_run().add_picture(caminho, width=Inches(largura_pol))
        legenda_p = self._paragrafo(self.m_legenda_fig,
                                    _runs_xml(f"Fig. {self._cont_fig}.  ", 16) + _runs_xml(legenda, 16))
        self._ajustar_ppr(legenda_p, '<w:spacing w:before="40" w:after="120"/><w:jc w:val="both"/>',
                          remover=("w:spacing", "w:jc"))
        self._anexar(legenda_p)
        return self._cont_fig

    def _quebra_secao(self, colunas):
        """Parágrafo vazio que encerra a seção atual (quebra contínua)."""
        sect = copy.deepcopy(self.sect_duas_colunas)
        tipo = sect.find(qn("w:type"))
        if tipo is None:
            tipo = parse_xml(f'<w:type {nsdecls("w")} w:val="continuous"/>')
            sect.insert(0, tipo)
        tipo.set(qn("w:val"), "continuous")
        if colunas == 1:
            sect.remove(sect.find(qn("w:cols")))
        p = parse_xml(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:before="0" w:after="0" w:line="120" '
                      'w:lineRule="exact"/><w:rPr><w:sz w:val="4"/></w:rPr></w:pPr></w:p>')
        p.find(qn("w:pPr")).append(sect)
        return self._anexar(p)

    def figura_larga(self, caminho, legenda, largura_pol=LARGURA_PAGINA_POL):
        """Figura ocupando as duas colunas (seção de 1 coluna entre quebras contínuas)."""
        self._quebra_secao(2)      # fecha o trecho em 2 colunas
        numero = self.figura(caminho, legenda, largura_pol)
        self._quebra_secao(1)      # fecha o trecho em 1 coluna e volta para 2 colunas
        return numero

    # -------------------------------------------------------- tabelas
    def proxima_tabela(self):
        return self._cont_tab + 1

    def tabela(self, titulo, cabecalho, linhas, larguras=None, negrito_linhas=()):
        """Tabela com a legenda acima (TABLE I. ...) no estilo do modelo."""
        self._cont_tab += 1
        romano = _romano(self._cont_tab)
        legenda = self._paragrafo(self.m_legenda_tab,
                                  _runs_xml(f"Tabela {romano}.  ", 16, small_caps=True)
                                  + _runs_xml(titulo, 16, small_caps=True))
        self._ajustar_ppr(legenda, '<w:keepNext/><w:spacing w:before="160" w:after="60"/>'
                                   '<w:ind w:left="0" w:right="0" w:firstLine="0"/>',
                          remover=("w:spacing", "w:ind", "w:tabs"))
        self._anexar(legenda)

        n_col = len(cabecalho)
        if larguras is None:
            primeira = int(LARGURA_COLUNA_TWIPS * 0.34)
            resto = (LARGURA_COLUNA_TWIPS - 120 - primeira) // (n_col - 1)
            larguras = [primeira] + [resto] * (n_col - 1)
        borda = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        bordas = "".join(borda.format(b) for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in larguras)

        def celula(texto, largura, negrito, alinhamento):
            return (f'<w:tc><w:tcPr><w:tcW w:w="{largura}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                    f'<w:p><w:pPr><w:pStyle w:val="TableParagraph"/><w:keepNext/>'
                    f'<w:spacing w:before="20" w:after="20"/><w:jc w:val="{alinhamento}"/></w:pPr>'
                    f'{_runs_xml(texto, 16, negrito_base=negrito)}</w:p></w:tc>')

        linhas_xml = ['<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(
            celula(c, w, True, "center") for c, w in zip(cabecalho, larguras)) + "</w:tr>"]
        for k, linha in enumerate(linhas):
            negrito = k in negrito_linhas
            linhas_xml.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
                celula(c, w, negrito, "left" if j == 0 else "center")
                for j, (c, w) in enumerate(zip(linha, larguras))) + "</w:tr>")
        xml = (f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{sum(larguras)}" w:type="dxa"/>'
               f'<w:jc w:val="center"/><w:tblBorders>{bordas}</w:tblBorders><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar>'
               f'<w:tblLook w:val="01E0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(linhas_xml)}</w:tbl>')
        self._anexar(parse_xml(xml))
        # pequeno espaço depois da tabela
        espaco = parse_xml(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:before="0" w:after="60"/>'
                           '<w:rPr><w:sz w:val="8"/></w:rPr></w:pPr></w:p>')
        self._anexar(espaco)
        return romano

    # ---------------------------------------------------- referências
    def titulo_referencias(self):
        p = self._paragrafo(self.m_corpo, _runs_xml("Referências", 20, small_caps=True))
        self._ajustar_ppr(p, '<w:keepNext/><w:spacing w:before="200" w:after="80"/><w:ind w:firstLine="0"/>'
                             '<w:jc w:val="center"/>', remover=("w:spacing", "w:ind", "w:jc"))
        self._anexar(p)

    def referencia(self, texto):
        p = self._paragrafo(self.m_referencia, _runs_xml(texto, 16))
        return self._anexar(p)

    # ------------------------------------------------------- finalizar
    def salvar(self, caminho):
        self._quebra_secao(2)  # encerra o corpo em 2 colunas
        self.doc.core_properties.language = "pt-BR"
        self.doc.save(caminho)


def _romano(n):
    valores = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    texto = ""
    for valor, simbolo in valores:
        while n >= valor:
            texto += simbolo
            n -= valor
    return texto

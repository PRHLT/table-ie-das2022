import glob, os, copy, pickle
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import cv2
# print(cv2.__version__)
from scipy.signal import convolve2d
import shapely
from shapely.geometry import LineString, Point
# construct the Laplacian kernel used to detect edge-like
# regions of an image


# construct the Sobel x-axis kernel
horizontal_kernel = np.array((
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]), dtype="int")
# horizontal_kernel = np.array((
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
#     [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), dtype="int")

# construct the Sobel y-axis kernel
vertical_kernel = np.array((
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]), dtype="int")


THICKNESS = 10
THRESHOLD = 100 #nº pixels
BASELINES = True
BINARY = False
DPI = 600
TWO_DIM = False
SHOW_IMG = False
COLS = True
ROWS = True
TABLE_BOX = True

iGRID_STEP = 33 # odd number is better

class TablePAGE():
    """
    Class for parse Tables from PAGE
    """

    def __init__(self, im_path, debug=False,
                 search_on=["TextLine"], hisClima=False, info_attributes=False, IoU=False, IoU_tl=[], IoU_path="", prod=False):
        """
        Set filename of inf file
        example : AP-GT_Reg-LinHds-LinWrds.inf
        :param fname:
        """
        self.im_path = im_path
        self.DEBUG_ = debug
        self.search_on = search_on
        self.min_cols = 10
        self.prod = prod

        self.parse()
        self.readingOrderTables = []
        self.offsetRows = {}
        self.maxRowSpan = 1
        tabla = False
        region = None
        self.IoU_tl = IoU_tl
        self.IoU = IoU
        
        if hisClima and not prod:
            # for every table
            if not info_attributes and not IoU: # PAGE GT
                for region in self.xmldoc.getElementsByTagName("TableRegion"):
                    if self.get_numColumns(region) < self.min_cols:
                        continue
                    rorder = self.get_ReadingOrder(region)
                    rows = []
                    #Get num rows
                    for i in region.childNodes:
                        if i.nodeName == 'TableCell':
                            row = int(i.attributes["row"].value)
                            rows.append(row)
                    max_row = max(rows)
                    self.readingOrderTables.append([rorder, max_row])
                
            elif IoU:
                self.xmldoc_gttable = minidom.parse(IoU_path)
                res_orders = {}
                for region in self.xmldoc_gttable.getElementsByTagName("TableRegion"):
                    if self.get_numColumns(region) < self.min_cols:
                        continue
                    rorder = self.get_ReadingOrder(region)
                    rows = []
                    #Get num rows
                    for i in region.childNodes:
                        if i.nodeName == 'TableCell':
                            row = int(i.attributes["row"].value)
                            rows.append(row)
                    max_row = max(rows)
                    self.readingOrderTables.append([rorder, max_row])
            else:
                
                res_orders = {}
                for region in self.xmldoc.getElementsByTagName("TextLine"):
                    id = region.attributes["id"].value
                    rorder = None
                    if "top" in id:
                        rorder = 1
                    elif "bot" in id:
                        rorder = 2
                    else:
                        continue
                    tabla = True
                    col, row, colspan, rowspan = self.get_custom_cellInfo(region)
                    rows = res_orders.get(rorder, [])
                    rows.append(row)
                    res_orders[rorder] = rows
                for rorder, rows in res_orders.items():
                    max_row = max(rows)
                    self.readingOrderTables.append([rorder, max_row])
            if region is not None or tabla: # Porque hay paginas en blanco
                    self.readingOrderTables.sort()
                    row_ant, offset_ant = self.readingOrderTables[0]
                    self.offsetRows[row_ant] = 0
                    for row, offset in self.readingOrderTables[1:]:
                        self.offsetRows[row] = offset_ant + 1
                        offset_ant = offset
                        row_ant = row

    def save_changes(self, ):
        with open(self.im_path,'w') as f:
            xml = self.xmldoc.toxml()
            f.write(xml)

    def get_ReadingOrder(self, region):
        if region is not None:
            rorder = region.attributes["custom"].value # "readingOrder {index:3;}"
            # print(rorder)
            rorder = int(rorder.split("index:")[-1].split(";")[0])
            return rorder
        else:
            return None
    
    def get_numColumns(self, region):
        cols = []
        if region is not None:
            for i in region.childNodes:
                if i.nodeName == 'TableCell':
                    i = int(i.attributes["col"].value)
                    cols.append(i)
            if not cols:
                return 0
            return max(cols)
        else:
            return None
    
    def get_custom_cellInfo(self, region):
        if region is not None:
            rorder = region.attributes["custom"].value # "readingOrder {index:3;}"
            
            try:
                col = int(rorder.split("col:")[-1].split(";")[0])
                row = int(rorder.split("row:")[-1].split(";")[0])
                colspan = int(rorder.split("colspan:")[-1].split(";")[0])
                rowspan = int(rorder.split("rowspan:")[-1].split(";")[0])
            except:
                col, row, colspan, rowspan = -1, -1, 1, 1
            return col, row, colspan, rowspan
        else:
            return None

    def get_daddy(self, node, searching="TextRegion"):
        while node.parentNode:
            node = node.parentNode
            if node.nodeName.strip() == searching:
                return node
        return None

    def get_text(self, node, nodeName="Unicode"):
        TextEquiv = None
        for i in node.childNodes:
            if i.nodeName == 'TextEquiv':
                TextEquiv = i
                break
        if TextEquiv is None:
            # print("No se ha encontrado TextEquiv en una región")
            return None

        for i in TextEquiv.childNodes:
            if i.nodeName == nodeName:
                try:
                    words = i.firstChild.nodeValue
                except:
                    words = ""
                return words

        return None
    
    def get_textLine_fromRegion(self, node, nodeName="Unicode"):
        TextEquiv = None
        for i in node.childNodes:
            if i.nodeName == 'TextLine':
                TextEquiv = i
                break
        if TextEquiv is None:
            return None
        return self.get_text(TextEquiv)
    
    def get_textLineswBL(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        baselines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            text = self.get_text(region)
            # try:
            coords_BL = self.get_baseline_fromTL(region)
            if coords_BL is None:
                continue
            coords_2 = []
            for i,j in coords_BL:
                coords_2.append(f'{str(i)},{str(j)}')
            c = " ".join(coords_2)

            # except:
            #     print(id, coords, text, coords_BL)
            #     exit()
            


            text_lines.append((coords, text, id, c))


        return text_lines

    def get_baseline_fromTL(self, node):
        TextEquiv = None
        for i in node.childNodes:
            if i.nodeName == 'Baseline':
                TextEquiv = i
                break
        if TextEquiv is None:
            return None
        return self.get_coords_BL(TextEquiv)

    def get_coords_BL(self, coords_element):
        """
        Devuelve las coordenadas de un elemento. Coords
        :param dom_element:
        :return: ((pos), (pos2), (pos3), (pos4)) es un poligono. Sentido agujas del reloj
        """
        coords = coords_element.attributes["points"].value
        coords = coords.split()
        coords_to_append = []
        for c in coords:
            x, y = c.split(",")
            coords_to_append.append((int(x), int(y)))
        return coords_to_append

    def get_TableRegion(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("TableRegion"):
            coords = self.get_coords(region)
            cells.append(coords)

        return cells

    def get_cells(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        cell_by_row = {}
        cell_by_col= {}
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            #TODO different tables
            coords = self.get_coords(region)

            row = int(region.attributes["row"].value)
            col = int(region.attributes["col"].value)
            cells.append((coords, col, row))

            cols = cell_by_col.get(col, [])
            cols.append(coords)
            cell_by_col[col] = cols

            rows = cell_by_row.get(row, [])
            rows.append(coords)
            cell_by_row[row] = rows

        return cells, cell_by_col, cell_by_row
    
    def get_cellsWithText(self,):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        cell_by_row = {}
        cell_by_col= {}
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            #TODO different tables
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            text = self.get_textLine_fromRegion(region)
            if not self.prod:
                row = int(region.attributes["row"].value)
                col = int(region.attributes["col"].value)
            else:
                row, col = -1, -1
            cells.append((coords, text, id, {
                    # 'DU_row': DU_row,
                    # 'DU_col': DU_col,
                    # 'DU_header': DU_header,
                    'row': row,
                    'col': col,
                    'fake': False
                }))

        return cells
    
    def get_cellsWithTextHisClima(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        cell_by_row = {}
        cell_by_col= {}
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            #TODO different tables
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            text = self.get_textLine_fromRegion(region)
            if not self.prod:
                tableregion = self.get_daddy(region, "TableRegion")
                rorder = self.get_ReadingOrder(tableregion)
                if rorder is None:
                    offset_row = 0
                else:
                    offset_row = self.offsetRows[rorder]
                row, col = -1,-1
                if region is not None:
                    row, col = int(region.attributes["row"].value), int(region.attributes["col"].value)
                    rowspan, colspan = int(region.attributes["rowSpan"].value), int(region.attributes["colSpan"].value)
                    # Sum offset row
                row += offset_row
                DU_row, DU_col, DU_header = row, col, 0

                if not offset_row and row == 0: # Solo tabla 1 - actualizamos rowSpan
                    self.maxRowSpan = max(self.maxRowSpan, rowspan)
                # Header
                if not offset_row: # Solo tabla 1
                    if row != -1 and row - self.maxRowSpan < 0:
                        DU_header = 1
            else:
                row, col = -1, -1
            cells.append((coords, text, id, {
                    # 'DU_row': DU_row,
                    # 'DU_col': DU_col,
                    'DU_header': DU_header,
                    'row': row,
                    'col': col,
                    'fake': False
                }))

        return cells
    
    def get_textLines_list(self, node):
        res = []
        for i in node.childNodes:
            if i.nodeName == 'TextLine':
                res.append(i.attributes["id"].value)
        return res

    def add_col_to_cell(self, dict_line_group):
        def most_common(lst):
            if not lst:
                return None
            return max(set(lst), key=lst.count)
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            row = int(region.attributes["row"].value)
            col = int(region.attributes["col"].value)
            tls = self.get_textLines_list(region)
            cols_tablecell = []
            for tl in tls:
                col_class = dict_line_group.get(tl, -1)
                cols_tablecell.append(col_class)
            col_num = most_common(cols_tablecell)
            # print(row, col, tls, cols_tablecell, col_num)
            if col_num is not None:
                if col != col_num:
                    print(row, col, tls, cols_tablecell, col_num)
                region.attributes["col"].value = str(col_num)
        



    def get_Separator(self, vertical=True):
        """
        Return all the GridSeparator in a PAGE (TranskribusDU - ABTTableGrid)
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("SeparatorRegion"):
            # orient_reg = int(region.attributes["orient"])
            orient_reg = "vertical" in region.attributes["orient"].value
            # if orient == orient_reg:
            if vertical == orient_reg :
                coords = self.get_coords(region)
                cells.append(coords)
        return cells

    def get_GridSeparator(self, orient=90):
        """
        Return all the GridSeparator in a PAGE (TranskribusDU - ABTTableGrid)
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("GridSeparator"):
            orient_reg = int(region.attributes["orient"].value)
            # orient_reg = "vertical" in region.attributes["orient"].value
            if orient == orient_reg:
            # if vertical == orient_reg :
                coords = self.get_coords(region)
                cells.append(coords)
        return cells

    def get_Baselines(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("Baseline"):
            coords = region.attributes["points"].value
            coords = coords.split()
            coords_to_append = []
            for c in coords:
                x, y = c.split(",")
                coords_to_append.append((int(x), int(y)))

            text_lines.append(coords_to_append)


        return text_lines



    def get_textLines(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines.append((coords, text))


        return text_lines
    
    def get_textLinesDict(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = {}
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines[id] = (coords, text)


        return text_lines
    
    def get_textLinesWithId(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines.append((coords, text, id))


        return text_lines
    
    def set_text(self, idLine, text):
        """
        Añadir debajo de "Baseline"
        Ej:
        <TextEquiv>
            <Unicode>" [cum.str nimb]</Unicode>
        </TextEquiv>        
        """
        
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            if id != idLine:
                continue
            TextEquiv = self.xmldoc.createElement("TextEquiv")

            Unicode = self.xmldoc.createElement("Unicode")
            text = self.xmldoc.createTextNode(text)
            Unicode.appendChild(text)
            TextEquiv.appendChild(Unicode)
            region.appendChild(TextEquiv)
            return


        # return text_lines

    def get_textLinesWithObject(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines.append((coords, text, region))


        return text_lines
    
    def get_width_coords(self, coords):
        min_x = min([x[0] for x in coords])
        max_x = max([x[0] for x in coords])
        return max_x - min_x

    def coords_blank_line(self, coords, width_m=0.3, height_m = 0.1):
        min_x = min([x[0] for x in coords])
        max_x = max([x[0] for x in coords])
        min_y = min([x[1] for x in coords])
        max_y = max([x[1] for x in coords])
        width = max_x - min_x
        height = max_y - min_y
        rest_w = int((width-(width*width_m)) / 2)
        min_x += rest_w
        max_x -= rest_w
        rest_h = int((height-(height*height_m)) / 2)
        min_y += rest_h
        max_y -= rest_h
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


    def get_textLinesFromCell(self, maxWidth=9999999, info_attributes=False, empty_cells=False):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            if not coords:
                continue
            width_line = self.get_width_coords(coords)
            if width_line > maxWidth:
                continue
            text = self.get_text(region)

            tablecell = self.get_daddy(region, "TableCell")
            tableregion = self.get_daddy(region, "TableRegion")
            rorder = self.get_ReadingOrder(tableregion)

            if not self.prod:
                tablecell = self.get_daddy(region, "TableCell")
                row, col = -1,-1
                rowspan, colspan = 0, 0
                if tablecell is not None:
                    row, col = int(tablecell.attributes["row"].value), int(tablecell.attributes["col"].value)
                    try:
                        rowspan, colspan = int(tablecell.attributes["rowSpan"].value), int(tablecell.attributes["colSpan"].value)
                    except Exception as e:
                        # print(f'fallo en id de linea {id}')
                        # raise e
                        rowspan, colspan = 1, 1
                # print(id, row, col)
                try:
                    DU_row, DU_col, DU_header = region.attributes["DU_row"].value, \
                                                region.attributes["DU_col"].value, \
                                                region.attributes["DU_header"].value
                except:
                    DU_row, DU_col, DU_header = row, col, 0

                text_lines.append((coords, text, id, {
                    'DU_row': DU_row,
                    'DU_col': DU_col,
                    'DU_header': DU_header,
                    'row': row,
                    'col': col,
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'fake': False,
                    'rorder': rorder,
                }))
            else:
                text_lines.append((coords, text, id, {
                    
                }))

        return text_lines
    
    def get_textLinesHisClima(self, maxWidth=9999999, info_attributes=False, empty_cells=False):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        max_row = 0
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            if not coords:
                continue
            width_line = self.get_width_coords(coords)
            if width_line > maxWidth:
                continue
            text = self.get_text(region)
            if not self.prod and not self.IoU: # PAGE
                ro = self.get_ReadingOrder(region)
                if not info_attributes:
                    tablecell = self.get_daddy(region, "TableCell")
                    tableregion = self.get_daddy(region, "TableRegion")
                    rorder = self.get_ReadingOrder(tableregion)
                else:
                    rorder = None
                    if "top" in id:
                        rorder = 1
                    elif "bot" in id:
                        rorder = 2
                if rorder is None:
                    offset_row = 0
                # elif rorder <= 1:
                else:
                    try:
                        offset_row = self.offsetRows[rorder]
                    except:
                        # No esta, es una tabla omitida
                        offset_row = 0
                row, col = -1,-1
                rowspan, colspan = 1, 1
                if not info_attributes:
                    if tablecell is not None:
                        row, col = int(tablecell.attributes["row"].value), int(tablecell.attributes["col"].value)
                        try:
                            rowspan, colspan = int(tablecell.attributes["rowSpan"].value), int(tablecell.attributes["colSpan"].value)
                        except:
                            rowspan, colspan = 1, 1
                else:
                    res_ = self.get_custom_cellInfo(region)
                    if res_ is not None:
                        col, row, colspan, rowspan = res_
                numColsTableRegion = self.get_numColumns(tableregion)
                if tablecell is not None and numColsTableRegion < self.min_cols:
                    row, col = -1,-1
                    offset_row = 0
                    rorder = None
                # Sum offset row
                row += offset_row
                DU_row, DU_col, DU_header = row, col, 0
                if not offset_row and row == 0: # Solo tabla 1 - actualizamos rowSpan
                    self.maxRowSpan = max(self.maxRowSpan, rowspan)
                # Header
                if not offset_row: # Solo tabla 1
                    if row != -1 and row - self.maxRowSpan < 0:
                        DU_header = 1
                # print(row, offset_row, " - ", DU_header)
                # if colspan != 1:
                #     print(" ********************************* ", colspan)
                #     exit()
                # else:
                #     print(colspan)
                text_lines.append((coords, text, id, {
                    'DU_row': DU_row,
                    'DU_col': DU_col,
                    'DU_header': DU_header,
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'row': row,
                    'col': col,
                    'fake': False,
                    "reading_order": ro
                }))
                max_row = max(max_row, row)
            elif self.IoU: # From another PAGE
                coords_np = np.array(coords)
                x1, y1, x2, y2 = min(coords_np[:,0]), min(coords_np[:,1]), max(coords_np[:,0]), max(coords_np[:,1])
                if not self.IoU_tl:
                    row, col, rowspan, colspan, rorder = -1, -1, 0, 0, None
                else:
                    iou, cell_gt, row, col, rowspan, colspan, rorder = match_IoU_TL([x1, y1, x2, y2], self.IoU_tl)
                # print(row, col)
                
                # if rorder is None:
                #     offset_row = 0
                # elif rorder <= 1:
                #     print(self.offsetRows)
                #     offset_row = self.offsetRows[rorder]
                # else:
                #     offset_row = 0
                # print(iou)

                # if self.get_numColumns(tablecell) < self.min_cols:
                #     row, col = -1,-1
                #     offset_row = 0
                #     rorder = None
                ro = rorder
                # row += offset_row
                DU_row, DU_col, DU_header = row, col, 0
                # if not offset_row: # Solo tabla 1
                if row != -1 and row < 2:
                        DU_header = 1
                text_lines.append((coords, text, id, {
                    'DU_row': DU_row,
                    'DU_col': DU_col,
                    'DU_header': DU_header,
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'row': row,
                    'col': col,
                    'fake': False,
                    "reading_order": ro
                }))
            else:
                text_lines.append((coords, text, id, {}))
        # if max_row != 0 and max_row != 27:
        #     print("Fallo en " , self.im_path)
        # exit()
        return text_lines
    
    def get_CellsHisClima(self, info_attributes=False, empty_cells=False):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        max_row = 0
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            if not coords:
                continue
            text = self.get_textLine_fromRegion(region) #TODO multi cell
            if not self.prod and not self.IoU: # PAGE
                if not info_attributes:
                    tableregion = self.get_daddy(region, "TableRegion")
                    rorder = self.get_ReadingOrder(tableregion)
                else:
                    rorder = None
                    if "top" in id:
                        rorder = 1
                    elif "bot" in id:
                        rorder = 2
                if rorder is None:
                    offset_row = 0
                # elif rorder <= 1:
                else:
                    try:
                        offset_row = self.offsetRows[rorder]
                    except:
                        # No esta, es una tabla omitida
                        offset_row = 0
                row, col = -1,-1
                rowspan, colspan = 1, 1
                if not info_attributes:
                    row, col = int(region.attributes["row"].value), int(region.attributes["col"].value)
                    try:
                        rowspan, colspan = int(region.attributes["rowSpan"].value), int(region.attributes["colSpan"].value)
                    except:
                        rowspan, colspan = 1, 1
                else:
                    res_ = self.get_custom_cellInfo(region)
                    if res_ is not None:
                        col, row, colspan, rowspan = res_
                numColsTableRegion = self.get_numColumns(tableregion)
                if numColsTableRegion < self.min_cols:
                    row, col = -1,-1
                    offset_row = 0
                    rorder = None
                # Sum offset row
                row += offset_row
                DU_row, DU_col, DU_header = row, col, 0
                if not offset_row and row == 0: # Solo tabla 1 - actualizamos rowSpan
                    self.maxRowSpan = max(self.maxRowSpan, rowspan)
                # Header
                if not offset_row: # Solo tabla 1
                    if row != -1 and row - self.maxRowSpan < 0:
                        DU_header = 1
                # print(row, offset_row, " - ", DU_header)
                text_lines.append((coords, text, id, {
                    'DU_row': DU_row,
                    'DU_col': DU_col,
                    'DU_header': DU_header,
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'row': row,
                    'col': col,
                    "reading_order": None,
                    'fake': False
                }))
                max_row = max(max_row, row)
        return text_lines
    
    def get_textLinesHisClima_Headers(self, maxWidth=9999999, info_attributes=False, empty_cells=False):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        max_row = 0
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            
            coords = self.get_coords(region)
            if not coords:
                continue
            width_line = self.get_width_coords(coords)
            if width_line > maxWidth:
                continue
            text = self.get_text(region)
            
            
            if not self.prod and not self.IoU: # PAGE
                ro = self.get_ReadingOrder(region)
                if not info_attributes:
                    tablecell = self.get_daddy(region, "TableCell")
                    tableregion = self.get_daddy(region, "TableRegion")
                
                    rorder = self.get_ReadingOrder(tableregion)
                else:
                    rorder = None
                    if "top" in id:
                        rorder = 1
                    elif "bot" in id:
                        rorder = 2
                if rorder is None:
                    offset_row = 0
                # elif rorder <= 1:
                else:
                    try:
                        offset_row = self.offsetRows[rorder]
                    except:
                        # No esta, es una tabla omitida
                        offset_row = 0
                row, col = -1,-1
                rowspan, colspan = 1, 1
                if not info_attributes:
                    if tablecell is not None:
                        row, col = int(tablecell.attributes["row"].value), int(tablecell.attributes["col"].value)
                        try:
                            rowspan, colspan = int(tablecell.attributes["rowSpan"].value), int(tablecell.attributes["colSpan"].value)
                        except:
                            rowspan, colspan = 1, 1
                else:
                    res_ = self.get_custom_cellInfo(region)
                    if res_ is not None:
                        col, row, colspan, rowspan = res_
                numColsTableRegion = self.get_numColumns(tableregion)
                if tablecell is not None and numColsTableRegion < self.min_cols:
                    row, col = -1,-1
                    offset_row = 0
                    rorder = None
                # Sum offset row
                row += offset_row
                DU_row, DU_col, DU_header = row, col, 0
                if not offset_row and row == 0: # Solo tabla 1 - actualizamos rowSpan
                    self.maxRowSpan = max(self.maxRowSpan, rowspan)
                # Header
                if not offset_row: # Solo tabla 1
                    if row != -1 and row - self.maxRowSpan < 0:
                        DU_header = 1
                # print(row, offset_row, " - ", DU_header)
                if DU_header:
                    text_lines.append((coords, text, id, {
                        'DU_row': DU_row,
                        'DU_col': DU_col,
                        'DU_header': DU_header,
                        'rowspan': rowspan,
                        'colspan': colspan,
                        'row': row,
                        'col': col,
                        'fake': False,
                        "reading_order": ro
                    }))
                    max_row = max(max_row, row)
            elif self.IoU: # From another PAGE
                coords_np = np.array(coords)
                x1, y1, x2, y2 = min(coords_np[:,0]), min(coords_np[:,1]), max(coords_np[:,0]), max(coords_np[:,1])
                if not self.IoU_tl:
                    row, col, rowspan, colspan, rorder = -1, -1, 0, 0, None
                else:
                    iou, cell_gt, row, col, rowspan, colspan, rorder = match_IoU_TL([x1, y1, x2, y2], self.IoU_tl)
                    # print(iou)
                if rorder is None:
                    offset_row = 0
                    # print("rorder None")
                elif rorder <= 1:
                    offset_row = self.offsetRows.get(rorder, 0)
                else:
                    offset_row = 0
                # print(rorder)
                ro = rorder
                # print(iou)

                # if self.get_numColumns(tablecell) < self.min_cols:
                #     row, col = -1,-1
                #     offset_row = 0
                #     rorder = None

                row += offset_row
                DU_row, DU_col, DU_header = row, col, 0
                if not offset_row: # Solo tabla 1
                    if row != -1 and row < 2:
                            DU_header = 1
                if DU_header:
                    text_lines.append((coords, text, id, {
                        'DU_row': DU_row,
                        'DU_col': DU_col,
                        'DU_header': DU_header,
                        'rowspan': rowspan,
                        'colspan': colspan,
                        'row': row,
                        'col': col,
                        'fake': False,
                        "reading_order": ro
                    }))
            else:
                text_lines.append((coords, text, id, {}))

        return text_lines

#    if DU_header:
#                 text_lines.append((coords, text, id, {
#                     'DU_row': DU_row,
#                     'DU_col': DU_col,
#                     'DU_header': DU_header,
#                     'row': row,
#                     'col': col,
#                     'rowspan': rowspan,
#                     'colspan': colspan,
#                     'fake': False,
#                     "reading_order": ro
#                 }))

    
    def get_TL_headers(self, info_attributes=False, maxWidth=9999999,):
        tls = self.get_textLinesHisClima_Headers(maxWidth=maxWidth, info_attributes=info_attributes)
        res = []
        for coords, text, id, dict_info in tls:
            row, col = dict_info['row'], dict_info['col']
            res.append([row, col, coords, text, id, dict_info])
        sorted(res, key=lambda element: (element[0], element[1]))
        # print(res)
        gts = {}
        tls = []
        for row, col, coords, text, id, dict_info in res:
            tls.append((coords, text, id, dict_info))
            colspan = dict_info['colspan']
            rowspan = dict_info['rowspan']
            if colspan == 1:
                gts[(row, col)] = []
            else:
                links = []
                for i in range(0, colspan):
                    links.append([row+1, col+i])
                gts[(row, col)] = links
        # for k,v in gts.items():
        #     print(k, v)
        # exit()
        return tls, gts
    
    def get_TL_headers_RO(self, info_attributes=False, maxWidth=9999999,):
        tls = self.get_textLinesHisClima_Headers(maxWidth=maxWidth, info_attributes=info_attributes)
        res = []
        # print(f" **** tls {tls}")
        for coords, text, id, dict_info in tls:
            row, col = dict_info['row'], dict_info['col']
            ro = dict_info['reading_order']
            res.append([row, col, coords, text, id, dict_info, ro])
        sorted(res, key=lambda element: (element[0], element[1]))
        # print(res)
        gts = {}
        tls = []
        tls_cells = {}
        for row, col, coords, text, id, dict_info, ro in res:
            tls.append((coords, text, id, dict_info))
            colspan = dict_info['colspan']
            rowspan = dict_info['rowspan']
            # print(colspan)
            if colspan == 1:
                gts[(row, col)] = []
            else:
                links = []
                for i in range(0, colspan):
                    links.append([row+1, col+i])
                gts[(row, col)] = links
            tl_cell = tls_cells.get((row, col), [])
            tl_cell.append([coords, text, id, dict_info, ro])
            if ro is None:
                print(tl_cell[-1])
            tls_cells[(row, col)] = tl_cell
        # print(f"gts {gts}")
        # exit()
        # print(f" **** tls_cells {tls_cells}")
        for k,v in tls_cells.items():
            
            sorted(v, key=lambda x: x[-1])
            # print(k, v)
            # [print(x[-1]) for x in v]
            tls_cells[k] = v
        # for k,v in gts.items():
        #     print(k, v)
        connections = {}
        # conexiones entre celdas
        for row, col, coords, text, id, dict_info, ro in res:
            conn = gts.get((row,col), [])
            line_to_connect = tls_cells.get((row,col))[-1]
            line_to_connect_id = line_to_connect[2]
            for row_j, col_j in conn:
                try:
                    line_to_connect_j = tls_cells.get((row_j, col_j))[0]
                except Exception as e:
                    print(f'Error with row {row_j} and col {col_j}')
                    continue
                line_to_connect_j_id = line_to_connect_j[2]
                connections_rc = connections.get(line_to_connect_id, [])
                connections_rc.append(line_to_connect_j_id)
                connections[line_to_connect_id] = connections_rc
        
        # conexiones dentro de la celda
        for k,v in tls_cells.items():
            last_tl = ""
            for coords, text, id, dict_info, ro in v:
                if last_tl:
                    connections_rc = connections.get(last_tl, [])
                    connections_rc.append(id)
                    connections[last_tl] = connections_rc
                last_tl = id
        return tls, gts, connections

    def get_coords(self, dom_element):
        """
        Devuelve las coordenadas de un elemento. Coords
        :param dom_element:
        :return: ((pos), (pos2), (pos3), (pos4)) es un poligono. Sentido agujas del reloj
        """
        coords_element = None
        for i in dom_element.childNodes:
            if i.nodeName == 'Coords':
                coords_element = i
                break
        if coords_element is None:
            print("No se ha encontrado coordenadas en una región")
            return None

        coords = coords_element.attributes["points"].value
        coords = coords.split()
        coords_to_append = []
        for c in coords:
            x, y = c.split(",")
            coords_to_append.append((int(x), int(y)))
        return coords_to_append
    
    def has_TextLine(self, dom_element):
        """
        
        """
        for i in dom_element.childNodes:
            if i.nodeName == 'TextLine':
                return True
        return False


    def parse(self):
        self.xmldoc = minidom.parse(self.im_path)

    def get_width(self):
        page = self.xmldoc.getElementsByTagName('Page')[0]
        return int(page.attributes["imageWidth"].value)

    def get_height(self):
        page = self.xmldoc.getElementsByTagName('Page')[0]
        return int(page.attributes["imageHeight"].value)


def get_all_xml(path, ext="xml"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names


def make_cells(cells, width, height):
    """
    make all the cells
    :param cells:
    :param width:
    :param height:
    :return:
    """
    drawing = np.zeros((height, width, 3), np.int32)
    for i in range(len(cells)):
        cell_,_,_ = cells[i]
        cell = np.array([cell_], dtype = np.int32)
        cv2.polylines(drawing, [cell],
             isClosed = True,
              color = (255,255,255),
              # color = 255,
              thickness = 10,)
    return drawing


def show_cells(drawing, title=""):
    """
    Show the image
    :return:
    """
    plt.title(title)
    plt.imshow(drawing)
    plt.show()
    plt.close()


def show_all_imgs(drawing, title="", redim=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(1, len(drawing))
    for i in range(len(drawing)):
        if redim is not None:
            a = resize(drawing[i], redim)
            print(a.shape)
            ax[i].imshow(a)
        else:
            ax[i].imshow(drawing[i])
    plt.savefig("aqui.eps", format='eps', dpi=900)
    plt.show( dpi=900)
    plt.close()

def show_all(img, cells, cols, rows, cols_processed, rows_processed, title="", dest=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(3, 2)
    # drawing -= drawing.min()
    # drawing /= drawing.max()
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(cells)
    ax[1, 0].imshow(cols)
    ax[1, 1].imshow(rows)
    ax[2, 0].imshow(cols_processed)
    ax[2, 1].imshow(rows_processed)
    if dest is not None:
        plt.savefig(dest, format='eps', dpi=DPI)
    else:
        if SHOW_IMG:
            plt.show()
    plt.close()

def show_all_with_lines(img, cells, cols, rows, cols_processed, rows_processed, lines, title="", dest=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(3, 3)
    # drawing -= drawing.min()
    # drawing /= drawing.max()
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(cells)
    ax[0, 2].imshow(lines)
    ax[1, 0].imshow(cols)
    ax[1, 1].imshow(rows)
    ax[2, 0].imshow(cols_processed)
    ax[2, 1].imshow(rows_processed)
    if dest is not None:
        plt.savefig(dest, format='eps', dpi=DPI)
    else:
        if SHOW_IMG:
            plt.show()
    plt.close()

def clarify_horizontal(bw):
    """
    Remove pixels from and image
    :param bw:
    :return:
    """
    WINDOW_SIZE = 1 # pixels
    THRESHOLD = np.max(bw) * 500
    h_line = 0
    # First, search if there is a line
    for i in range(WINDOW_SIZE, bw.shape[0], WINDOW_SIZE):
        windowed = bw[i-WINDOW_SIZE:i, :]
        val_window = windowed.sum().sum()
        if val_window < THRESHOLD:
            h_line += 1
            bw[i - WINDOW_SIZE:i, :] = 0
    return bw

def clarify_vertical(bw):
    """
    Remove pixels from and image
    :param bw:
    :return:
    """
    WINDOW_SIZE = 1 # pixels
    THRESHOLD = np.max(bw) * 100
    h_line = 0
    # First, search if there is a line
    for i in range(WINDOW_SIZE, bw.shape[1], WINDOW_SIZE):
        windowed = bw[:, i-WINDOW_SIZE:i]
        val_window = windowed.sum().sum()
        if val_window < THRESHOLD:
            h_line += 1
            bw[:, i - WINDOW_SIZE:i] = 0
    return bw

def get_axis_from_points(coords_, n=2, axis=1, func="max"):
    """

    :param coords:
    :param n: number of points to  get
    :param axis: y=1, x=0
    :return:
    """
    points = []
    coords = copy.copy(coords_)
    if type(coords[0]) == list:
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in coords])
            else:
                m = np.argmin([x[axis] for x in coords])
            points.append(coords[m])
            # print(coords)
            # print(m)
            del coords[m]
    else:

        aux = [[x[0],x[1]] for x in coords]
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in aux])
            else:
                m = np.argmin([x[axis] for x in aux])
            points.append(aux[m])
            del aux[m]

    return points



def get_lines(tables, img, cells, cell_by_col, cell_by_row, height, width, size=None, title="", dest=None, baseLines=None):
    """
    Get the lines for every row and column. Separe horitzontal and vertical
    :param cells:
    :param cell_by_col:
    :param cell_by_row:
    :param height:
    :param width:
    :return:
    """
    print(title)
    width_img, height_img,_ = img.shape
    img_cols = np.zeros((height, width), np.int32)
    img_rows = np.zeros((height, width), np.int32)
    lines_h = np.zeros((height, width), np.int32)
    lines_w = np.zeros((height, width), np.int32)
    lines_cells = np.zeros((height, width), np.int32)
    lines_box = np.zeros((height, width), np.int32)
    # n_cols = len(list(cell_by_col.keys()))
    # n_rows = len(list(cell_by_row.keys()))
    # print("A total of {} cols and {} rows on an image of {} height and {} width".format(n_cols, n_rows,  height, width))

    # [for horizontal]
    for coords, _, _ in cells:
        points = []
        points.extend(get_axis_from_points(coords, axis=1, func="max"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_h, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

        points = []
        points.extend(get_axis_from_points(coords, axis=1, func="min"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_h, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    lines_h = convolve2d(lines_h, np.ones((5,1),np.uint8), 'same')
    # [for vertical]
    # for i in list(cell_by_col.keys()):
    #     coords = cell_by_col[i]
    for coords,_,_ in cells:

        points = []
        points.extend(get_axis_from_points(coords, axis=0, func="max"))

        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_w, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
        points = []
        points.extend(get_axis_from_points(coords, axis=0, func="min"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_w, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    """CELLS"""
    box = []
    for cell,_,_ in cells:
        box.extend(cell)

        cell = np.array([cell], dtype=np.int32)

        cv2.polylines(lines_cells, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    """End cells"""


    """Table shape
    This piece of code detects the rectangular or not rectangular shape of the table :)
    """
    coco_labels = []
    for coords in tables:

        x = [p[0] for p in coords]
        y = [p[1] for p in coords]
        centroid = [(sum(x) / len(coords))/height_img, (sum(y) / len(coords))/width_img]
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        width_ = (max_y - min_y) / width_img
        height_ = (max_x - min_x) / height_img
        coco_labels.append((centroid[0], centroid[1], width_, height_))
        cell = np.array([coords], dtype=np.int32)
        cv2.polylines(lines_box, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    # show_cells(lines_cells, title)
    # show_cells(img, title)

    """End table shape"""

    """textlines"""
    if baseLines is not None:
        textLinesDrawing = np.zeros((height, width), np.int32)
        for coords in baseLines:
            # coords, text = textline
            coords = np.array([coords], dtype=np.int32)
            cv2.polylines(textLinesDrawing, [coords],
                          isClosed=False,
                          color=(255, 255, 255),
                          # color = 255,
                          thickness=THICKNESS, )

    # show_cells(lines_cells, title)
    # show_cells(img, title)
    """End textlines"""
    lines_w = convolve2d(lines_w, np.ones((1,5),np.uint8), 'same')
    lines_w = np.where(lines_w > 0, 1, 0)
    lines_h = np.where(lines_h > 0, 1, 0)
    lines_h = clarify_horizontal(lines_h)
    # lines_w = clarify_vertical(lines_w)
    #Resize
    if resize is not None:
        img = resize(img, size=size)
        lines_cells = resize(lines_cells, size=size)
        lines_h = resize(lines_h, size=size)
        lines_w = resize(lines_w, size=size)
        lines_box = resize(lines_box, size=size)
        if baseLines is not None:
            textLinesDrawing = resize(textLinesDrawing, size=size)
    # show_cells(lines_h, title)
    # show_cells(lines_w, title)

    # exit()
    if not TWO_DIM:
        cols = get_1dim(lines_h, axis=1)
        rows = get_1dim(lines_w, axis=0)
    else:
        cols = get_2dim(lines_h, axis=1)
        rows = get_2dim(lines_w, axis=0)

    cols_img = create_img(cols, size[0], reverse=True)
    rows_img = create_img(rows, size[1])
    # show_cells(cols_img, title)
    # show_cells(rows_img, title)

    lines_cells = np.where(lines_cells > 127, 1, 0)

    if baseLines is not None:
        show_all_with_lines(img, lines_cells, lines_h,
                            lines_w, cols_img, rows_img, lines=textLinesDrawing,
                            title=title, dest=dest)
        # lines_cells = img_to_onehot(lines_cells, color_dict)
        return lines_cells, cols, rows, textLinesDrawing, lines_box, lines_h, lines_w, coco_labels
    else:
        show_all(resize(img, size=size), resize(lines_cells, size=size), lines_h, lines_w,
                 cols_img, rows_img, title=title, dest=dest)
        # lines_cells = img_to_onehot(lines_cells, color_dict)
        return lines_cells, cols, rows, lines_box, lines_h, lines_w, coco_labels

def get_Separators(img, page, height, width, size=None, title="", dest=None):
    """
    Get the lines for every row and column. Separe horitzontal and vertical
    :param cells:
    :param cell_by_col:
    :param cell_by_row:
    :param height:
    :param width:
    :return:
    """
    print(title)
    width_img, height_img,_ = img.shape
    img_cols = np.zeros((height, width), np.int32)
    img_rows = np.zeros((height, width), np.int32)

    # n_cols = len(list(cell_by_col.keys()))
    # n_rows = len(list(cell_by_row.keys()))
    # print("A total of {} cols and {} rows on an image of {} height and {} width".format(n_cols, n_rows,  height, width))

    cols = page.get_Separator(vertical=True)
    print(len(cols))
    # cols = page.get_GridSeparator(orient=90)
    for coords in cols:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_cols, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    rows = page.get_Separator(vertical=False)
    # rows =  page.get_GridSeparator(orient=0)
    for coords in rows:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )


    return img_cols, img_rows, cols, rows

def create_img(cols, size, reverse=False):
    if type(cols) is not tuple:
        if not reverse:
            a =  np.array([cols]*size)
        else:
            a =  np.array([cols]*size).T
    else:
        cols_, starts = cols
        a = np.zeros((len(cols_), size))
        for i in range(len(cols_)):
            start = starts[i]
            num_pix = cols_[i]
            nums = int((num_pix*size))
            a[i, start:start+nums] = 1
        if not reverse:
            a = a.T
    return a

def get_1dim(bw, axis=0):
    """
    get the result for the split model
    :param arr:
    :return:
    """
    if BINARY:
        threshold = THRESHOLD * np.max(bw)
        bw = bw.sum(axis=axis)
        bw = np.where(bw > threshold, 1, 0)
    else:
        div = bw.shape[axis]
        bw = bw.sum(axis=axis)/div
    return bw


def get_2dim(bw, axis=0):
    """
    get the result for the split model
    :param arr:
    :return:
    """
    div = bw.shape[axis]
    sum_pix = bw.sum(axis=axis)/div
    starts = np.argmax(bw, axis)
    return (sum_pix, starts)

def resize(img, size=(1024,512)):
    return cv2.resize(img.astype('float32'), size).astype('int32')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_cocofile(data, fname):
    with open(fname, 'w') as handle:
        c = 0
        for x_center, y_center, width, height in data:
            handle.write("{} {} {} {} {}".format(c, x_center, y_center, width, height))

def load_image(path, size):
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"
    image = cv2.imread(p)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image2, size=size)
    image = (image.transpose((2, 0, 1))).astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, image2

color_dict = {0: 0,
              1: 1,}

def img_to_onehot(img, color_dict):
    num_classes = len(color_dict)
    shape = img.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = (img.flatten() == color_dict[i]).reshape(shape[:2])
    return arr

def onehot_to_img(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2] )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

""" From DUTranskribus"""
def getGridBB(iPageWidth, iPageHeight):
        xmin = 0
        xmax = (iPageWidth // iGRID_STEP) * iGRID_STEP
        ymin = 0
        ymax = (iPageHeight // iGRID_STEP) * iGRID_STEP
        return xmin, ymin, xmax, ymax

def iterGridHorizontalLines(iPageWidth, iPageHeight):
    """
    Coord of the horizontal lines of the grid, given the page size (pixels)
    Return iterator on the (x1,y1, x2,y2)
    """
    xmin, ymin, xmax, ymax = getGridBB(iPageWidth, iPageHeight)
    # print("grid horizontal liines: ", end=" ")
    # print(xmin, ymin, xmax, ymax)
    res = []
    for y in range(ymin, ymax+1, iGRID_STEP):
        for i in range(-70,70,70):
            res.append( [(xmin, y),(xmax, min(max(0,y+i), iPageHeight))])
    return res

def create_skewed_lines(width, height, baselines):
    xmin, ymin, xmax, ymax = getGridBB(width, height)
    img_rows = np.zeros((height, width), np.int32)
    # rows = [
    #     [(xmin, 0), (xmax, 0)],
    #     [(xmin, 0), (xmax, 30)],
    #     [(xmin, 0), (xmax, 70)],
    #     [(xmin, 0), (xmax, 500)],
    #     [(xmin, 200), (xmax, 1000)],
    #     [(xmin, 0), (xmax, 1500)],
    #     [(xmin, 0), (xmax, 2000)],
    # ]
    all_edges = []
    for coords in baseLines:
        for i in range(0,len(coords)):
            if i+1 < len(coords):
                all_edges.append([coords[i], coords[i+1]])
    # print(all_edges)
    rows = iterGridHorizontalLines(width, height)
    print("Pair of coords: {} # of BL: {}".format(len(all_edges),len(baseLines)))
    print("Rows : {}  ".format(len(rows)))
    rows_results = []
    for coords in rows:
        # coords, text = textline
        line1 = LineString(coords)
        intersections = [line1.intersection(LineString(pair)) for pair in all_edges]
        if any(intersections):
            continue
        rows_results.append(coords)
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    print("rows_results : {}  ".format(len(rows_results)))

    for coords in baseLines:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    # show_cells(img_rows)

    return rows_results, img_rows

def get_line_min_distance(coord_line, rows):
    pass

def label_separatos(separators, gt):
    """
    0 -> False, not a true separator
    1 -> True, a separator
    :param separators:
    :param gt:
    :return:
    """
    res = []
    res_check = np.zeros(len(separators))
    for sep in separators:
        res.append([sep, 0])
    for sep in gt:
        distances = [distance(sep, x) for x in separators]
        candidate = np.argmin(distances)
        res[candidate][1] = 1
        res_check[candidate] = 1
    print("A total of {} true candidades with {} on gt".format(np.sum(res_check), len(gt)))
    return res

def distance(line1, line2):
    """
    [(xmin, 0), (xmax, 0)],
    [(xmin, 0), (xmax, 30)],
    :param line1:
    :param line2:
    :return:
    """
    # return abs(line1[0][0] - line2[0][0]) + abs(line1[0][1] - line2[0][1]) + \
    #         abs(line1[1][0] - line2[1][0]) + abs(line1[1][1] - line2[1][1])
    return abs(line1[0][1] - line2[0][1]) + \
          abs(line1[1][1] - line2[1][1])


def match_IoU_TL(hyp_BB, tls_GT):
    res = []
    hyp_BB = np.array(hyp_BB)
    # print("-> ", hyp_BB, cells_GT)
    for i, (coords,text,id_linea,info) in enumerate(tls_GT):
        coords = np.array(coords)
        coords = np.array([coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()])
        iou = IoU(hyp_BB, coords)
        row, col = info.get('row', -1), info.get('col', -1)
        rowSpan, colSpan = info.get('rowspan', 1), info.get('colspan', 1)
        # if colSpan != 1:
        #     print(colSpan)
        rorder = info.get('rorder', info.get('reading_order')) #  info.get('reading_order')
        if rorder is None:
            # print(id_linea, f"{rorder} --> 0 rorder")
            # print()
            # exit()
            rorder = 0
        res.append([iou, i, row, col, rowSpan, colSpan, rorder])
    # print(res)
    # print("----------")
    res.sort()
    # print(res)
    # exit()
    return res[-1]

def IoU(box1: np.ndarray, box2: np.ndarray):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio
